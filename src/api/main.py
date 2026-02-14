import os
import pandas as pd
import joblib
from flask import Flask, jsonify, request
import logging
import traceback
from preprocessing.pipeline import build_pipeline, save_pipeline
from training.trainer import ModelTrainer
from preprocessing.components import get_feature_lists

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AcademicRiskApp:
    """
    Main application class for the Academic Risk Prediction System.
    Acts as the orchestrator for training, inference, and API serving.
    """

    def __init__(self):
        self.app = Flask(__name__)
        self.setup_routes()
        self.pipeline = None

        # Load production model at startup
        self.model_path = os.path.join("models", "production", "model.joblib")
        self.model = self.load_model()

        # Get expected features for validation
        self.numeric_features, self.categorical_features = get_feature_lists()

        # Define derived features that should NOT be expected in raw input
        derived_features = ['IS_NEW_STUDENT']

        # Filter derived features from required input columns
        input_numeric = [f for f in self.numeric_features if f not in derived_features]
        input_categorical = [f for f in self.categorical_features if f not in derived_features]

        # Add raw features that are necessary for derivation
        self.required_columns = input_numeric + input_categorical + ['SINALIZADOR_INGRESSANTE']

    def load_model(self):
        """
        Safely loads the production model.
        """
        if os.path.exists(self.model_path):
            try:
                model = joblib.load(self.model_path)
                logger.info(f"Loaded production model from {self.model_path}")
                return model
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                return None
        else:
            logger.warning(f"No production model found at {self.model_path}")
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
        """
        Defines the Flask API endpoints.
        """

        @self.app.route('/health', methods=['GET'])
        def health():
            status = 'healthy' if self.model is not None else 'degraded (no model loaded)'
            return jsonify({'status': status, 'message': 'Academic Risk API is running'}), 200

        @self.app.route('/predict', methods=['POST'])
        def predict():
            """
            Inference endpoint.
            Expects a JSON payload with a list of records.
            """
            if self.model is None:
                # Try reloading
                self.model = self.load_model()
                if self.model is None:
                    return jsonify({'status': 'error', 'message': 'No model loaded for inference'}), 503

            try:
                data = request.get_json()
                if not data:
                    return jsonify({'status': 'error', 'message': 'Empty payload'}), 400

                # Handle single record or list of records
                if isinstance(data, dict):
                    data = [data]

                df = pd.DataFrame(data)

                # Validation
                is_valid, missing = self.validate_input(df)
                if not is_valid:
                    return jsonify({
                        'status': 'error',
                        'message': f'Missing required columns: {missing}'
                    }), 400

                # Inference
                # predict_proba returns [prob_class_0, prob_class_1]
                # We want prob_class_1 (risk)
                if hasattr(self.model, "predict_proba"):
                    probabilities = self.model.predict_proba(df)[:, 1]
                else:
                    probabilities = [None] * len(df)

                predictions = self.model.predict(df)

                results = []
                for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                    results.append({
                        "id": i,  # Or use an ID from input if available
                        "risk_prediction": int(pred),
                        "risk_probability": float(prob) if prob is not None else None,
                        "risk_label": "High Risk" if pred == 1 else "Low Risk"
                    })

                return jsonify({
                    'status': 'success',
                    'count': len(results),
                    'predictions': results
                }), 200

            except Exception as e:
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
            return self.train(data_path)

    @staticmethod
    def train(data_path: str = 'data/raw/PEDE_PASSOS_DATASET_FIAP.csv'):
        """
        Triggers the training process via the ModelTrainer.
        """
        try:
            trainer = ModelTrainer()
            best_model_name, best_score = trainer.train_and_evaluate(data_path)
            return jsonify({
                'status': 'success',
                'best_model': best_model_name,
                'recall_score': best_score,
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

    def run_server(self, host='0.0.0.0', port=5000):
        """
        Starts the Flask API server.
        """
        logger.info(f"Starting API server on {host}:{port}")
        self.app.run(host=host, port=port)


if __name__ == "__main__":
    app = AcademicRiskApp()
    app.run_server()
