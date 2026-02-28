import os
import time
import pandas as pd
import joblib
from flask import Flask, jsonify, request, Response, send_file
import logging
import traceback
import json
from datetime import datetime
from preprocessing.pipeline import build_pipeline, save_pipeline
from training.trainer import ModelTrainer
from api.training_params import DEFAULT_TRAINING_CONFIG, parse_training_params
from preprocessing.components import get_feature_lists
from utils.logger import setup_logger
from monitoring.drift import DriftDetector
from monitoring.metrics import (
    MODEL_INFO, MODEL_LOADED,
    PREDICTION_REQUEST_COUNT, PREDICTION_LATENCY, PREDICTION_PROBABILITY,
    PREDICTIONS_HIGH_RISK, PREDICTIONS_LOW_RISK,
    TRAINING_REQUESTS, TRAINING_BEST_SCORE,
    DRIFT_CHECK_COUNT, DRIFT_DETECTED, DRIFT_CHECK_DURATION,
    HTTP_REQUEST_COUNT, HTTP_REQUEST_LATENCY,
    metrics_response,
)

logger = setup_logger("api")


class AcademicRiskApp:
    """
    Main application class for the Academic Risk Prediction System.
    Acts as the orchestrator for training, inference, and API serving.
    """

    def __init__(self):
        self.app = Flask(__name__)
        self.swagger_path = os.path.join(os.path.dirname(__file__), "swagger.yml")
        self.swagger_ui_path = os.path.join(os.path.dirname(__file__), "docs", "swagger_ui.html")
        self.setup_routes()
        self.pipeline = None

        self.model_path = os.getenv("MODEL_PATH", os.path.join("models", "production", "model.joblib"))
        self.data_path = os.getenv("DATA_PATH", "data/raw/PEDE_PASSOS_DATASET_FIAP.csv")
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        
        logging.getLogger().setLevel(self.log_level)

        self.model = self.load_model()

        self.numeric_features, self.categorical_features = get_feature_lists()

        derived_features = ['IS_NEW_STUDENT']
        input_numeric = [f for f in self.numeric_features if f not in derived_features]
        input_categorical = [f for f in self.categorical_features if f not in derived_features]
        self.required_columns = input_numeric + input_categorical + ['SINALIZADOR_INGRESSANTE']

        MODEL_INFO.info({
            "model_path": self.model_path,
            "environment": os.getenv("ENVIRONMENT", "production"),
        })

        self._register_request_hooks()

    def _register_request_hooks(self):
        """Register before/after request hooks for HTTP-level metrics."""
        @self.app.before_request
        def _start_timer():
            request._start_time = time.perf_counter()

        @self.app.after_request
        def _record_metrics(response):
            if request.path == "/metrics":
                return response
            elapsed = time.perf_counter() - getattr(request, "_start_time", time.perf_counter())
            endpoint = request.path
            HTTP_REQUEST_COUNT.labels(
                method=request.method,
                endpoint=endpoint,
                status_code=response.status_code,
            ).inc()
            HTTP_REQUEST_LATENCY.labels(
                method=request.method,
                endpoint=endpoint,
            ).observe(elapsed)
            return response

    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                model = joblib.load(self.model_path)
                logger.info(f"Loaded production model from {self.model_path}")
                MODEL_LOADED.set(1)
                return model
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                MODEL_LOADED.set(0)
                return None
        else:
            logger.warning(f"No production model found at {self.model_path}")
            MODEL_LOADED.set(0)
            return None

    def validate_input(self, df):
        """
        Validates that the input DataFrame contains necessary columns.
        Returns (is_valid, missing_columns)
        """
        missing = [col for col in self.required_columns if col not in df.columns]
        if missing:
            return False, missing
        return True, []

    def setup_routes(self):
        @self.app.route('/swagger.yml', methods=['GET'])
        def swagger_spec():
            return send_file(self.swagger_path, mimetype='application/yaml')

        @self.app.route('/docs', methods=['GET'])
        def swagger_ui():
            return send_file(self.swagger_ui_path, mimetype='text/html')

        @self.app.route('/metrics', methods=['GET'])
        def prometheus_metrics():
            body, status, headers = metrics_response()
            return Response(body, status=status, headers=headers)

        @self.app.route('/health', methods=['GET'])
        def health():
            status = 'healthy' if self.model is not None else 'degraded (no model loaded)'
            return jsonify({'status': status, 'message': 'Academic Risk API is running'}), 200

        @self.app.route('/predict', methods=['POST'])
        def predict():
            if self.model is None:
                self.model = self.load_model()
                if self.model is None:
                    PREDICTION_REQUEST_COUNT.labels(status="error", risk_label="none").inc()
                    return jsonify({'status': 'error', 'message': 'No model loaded for inference'}), 503

            start = time.perf_counter()
            try:
                data = request.get_json()
                if not data:
                    PREDICTION_REQUEST_COUNT.labels(status="error", risk_label="none").inc()
                    return jsonify({'status': 'error', 'message': 'Empty payload'}), 400

                if isinstance(data, dict):
                    data = [data]

                df = pd.DataFrame(data)

                is_valid, missing = self.validate_input(df)
                if not is_valid:
                    PREDICTION_REQUEST_COUNT.labels(status="error", risk_label="none").inc()
                    return jsonify({
                        'status': 'error',
                        'message': f'Missing required columns: {missing}'
                    }), 400

                if hasattr(self.model, "predict_proba"):
                    probabilities = self.model.predict_proba(df)[:, 1]
                else:
                    probabilities = [None] * len(df)

                predictions = self.model.predict(df)

                elapsed = time.perf_counter() - start
                PREDICTION_LATENCY.observe(elapsed)

                results = []
                for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                    label = "High Risk" if pred == 1 else "Low Risk"
                    results.append({
                        "id": i,
                        "risk_prediction": int(pred),
                        "risk_probability": float(prob) if prob is not None else None,
                        "risk_label": label,
                    })

                    PREDICTION_REQUEST_COUNT.labels(status="success", risk_label=label).inc()

                    if prob is not None:
                        PREDICTION_PROBABILITY.observe(float(prob))

                    if pred == 1:
                        PREDICTIONS_HIGH_RISK.inc()
                    else:
                        PREDICTIONS_LOW_RISK.inc()

                for res in results:
                    logger.info("prediction_made", extra={
                        "event": "prediction",
                        "input_id": res["id"],
                        "prediction": res["risk_prediction"],
                        "probability": res["risk_probability"],
                        "latency_ms": round(elapsed * 1000, 2),
                        "timestamp": datetime.now().isoformat(),
                    })

                return jsonify({
                    'status': 'success',
                    'count': len(results),
                    'predictions': results
                }), 200

            except Exception as e:
                PREDICTION_REQUEST_COUNT.labels(status="error", risk_label="none").inc()
                logger.error(f"Prediction failed: {e}")
                traceback.print_exc()
                return jsonify({'status': 'error', 'message': str(e)}), 500

        @self.app.route('/pipeline/run', methods=['POST'])
        def run_pipeline():
            """
            Trigger the preprocessing pipeline execution.
            """
            try:
                body = request.get_json(silent=True) or {}
                data_path = body.get('data_path', 'data/raw/PEDE_PASSOS_DATASET_FIAP.csv')
                self.run_preprocessing_pipeline(data_path)
                return jsonify({'status': 'success', 'message': 'Pipeline executed successfully'}), 200
            except Exception as e:
                logger.error(f"Pipeline execution failed: {str(e)}")
                return jsonify({'status': 'error', 'message': str(e)}), 500

        @self.app.route('/train', methods=['POST'])
        def train_endpoint():
            body = request.get_json(silent=True) or {}
            data_path = body.get('data_path', 'data/raw/PEDE_PASSOS_DATASET_FIAP.csv')
            params_payload = body.get('training_params', {})
            training_config, validation_errors = parse_training_params(params_payload)

            if validation_errors:
                TRAINING_REQUESTS.labels(status="error").inc()
                return jsonify({
                    'status': 'error',
                    'message': 'Invalid training parameters',
                    'errors': validation_errors
                }), 400

            start = time.perf_counter()
            result = self.train(data_path, training_config)
            elapsed = time.perf_counter() - start
            response, status_code = result
            if status_code == 200:
                TRAINING_REQUESTS.labels(status="success").inc()
                resp_data = response.get_json()
                best_model = resp_data.get("best_model", "unknown")
                best_score = resp_data.get("recall_score", 0)
                TRAINING_BEST_SCORE.labels(
                    metric_name="recall", model_type=best_model
                ).set(best_score)
            else:
                TRAINING_REQUESTS.labels(status="error").inc()
            return result

        @self.app.route('/monitoring/drift', methods=['POST'])
        def check_drift():
            start = time.perf_counter()
            try:
                body = request.get_json(silent=True) or {}
                current_data_path = body.get('current_data_path', 'data/raw/PEDE_PASSOS_DATASET_FIAP.csv')
                reference_data_path = body.get('reference_data_path', 'data/raw/PEDE_PASSOS_DATASET_FIAP.csv')

                if not os.path.exists(current_data_path):
                    DRIFT_CHECK_COUNT.labels(status="error").inc()
                    return jsonify({'status': 'error', 'message': f'Current data file not found: {current_data_path}'}), 404

                detector = DriftDetector(reference_data_path, current_data_path)
                result = detector.run_drift_check()

                elapsed = time.perf_counter() - start
                DRIFT_CHECK_DURATION.observe(elapsed)
                DRIFT_CHECK_COUNT.labels(status="success").inc()
                DRIFT_DETECTED.set(1 if result['drift_detected'] else 0)

                return jsonify({
                    'status': 'success',
                    'drift_detected': result['drift_detected'],
                    'report_path': result['html_report']
                }), 200

            except Exception as e:
                DRIFT_CHECK_COUNT.labels(status="error").inc()
                logger.error(f"Drift check failed: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500

    @staticmethod
    def train(
        data_path: str = 'data/raw/PEDE_PASSOS_DATASET_FIAP.csv',
        training_config: dict | None = None
    ):
        try:
            trainer = ModelTrainer(config=training_config or DEFAULT_TRAINING_CONFIG.copy())
            best_model_name, best_score = trainer.train_and_evaluate(data_path)
            return jsonify({
                'status': 'success',
                'best_model': best_model_name,
                'recall_score': best_score,
                'experiment_name': trainer.experiment_name,
                'training_config': trainer.config,
                'message': 'Training completed successfully'
            }), 200
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500

    def run_preprocessing_pipeline(self, data_path: str):
        """
        Orchestrates the preprocessing pipeline:
        1. Loads data
        2. Builds pipeline
        3. Fits and transforms data
        4. Saves pipeline artifact
        """
        logger.info(f"Starting preprocessing pipeline with data from: {data_path}")

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found at {data_path}")

        try:
            # Load Data
            df = pd.read_csv(data_path, delimiter=';')
            logger.info(f"Data loaded. Shape: {df.shape}")

            # Build Pipeline
            self.pipeline = build_pipeline()

            # Execute Pipeline
            # Note: The cleaner step may reduce the number of rows
            X_transformed = self.pipeline.fit_transform(df)
            logger.info(f"Pipeline executed. Transformed data shape: {X_transformed.shape}")

            # Save Artifact
            save_pipeline(self.pipeline)
            logger.info("Pipeline artifact saved successfully.")

            return X_transformed

        except Exception as e:
            logger.error(f"Error during pipeline execution: {e}")
            traceback.print_exc()
            raise e

    def run_server(self, host='0.0.0.0', port=None):
        """
        Starts the Flask API server.
        """
        if port is None:
            port = int(os.getenv("PORT", 5000))
            
        logger.info(f"Starting API server on {host}:{port}")
        self.app.run(host=host, port=port)


if __name__ == "__main__":
    app = AcademicRiskApp()
    app.run_server()
