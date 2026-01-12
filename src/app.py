import os
import pandas as pd
from flask import Flask, jsonify, request
import logging
import traceback
from preprocessing.pipeline import build_pipeline, save_pipeline

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
        self.model = None  # Placeholder for Phase 3

    def setup_routes(self):
        """
        Defines the Flask API endpoints.
        """

        @self.app.route('/health', methods=['GET'])
        def health():
            return jsonify({'status': 'ok', 'message': 'Academic Risk API is running'}), 200

        @self.app.route('/pipeline/run', methods=['POST'])
        def run_pipeline():
            """
            Trigger the preprocessing pipeline execution.
            """
            try:
                data_path = request.json.get('data_path', 'data/raw/PEDE_PASSOS_DATASET_FIAP.csv')
                self.run_preprocessing_pipeline(data_path)
                return jsonify({'status': 'success', 'message': 'Pipeline executed successfully'}), 200
            except Exception as e:
                logger.error(f"Pipeline execution failed: {str(e)}")
                return jsonify({'status': 'error', 'message': str(e)}), 500

        @self.app.route('/train', methods=['POST'])
        def train_endpoint():
            data_path = request.json.get('data_path', 'data/raw/PEDE_PASSOS_DATASET_FIAP.csv')
            return self.train(data_path)

    @staticmethod
    def train(data_path: str = 'data/raw/PEDE_PASSOS_DATASET_FIAP.csv'):
        """
        Triggers the training process via the ModelTrainer.
        """
        try:
            from training.trainer import ModelTrainer
            trainer = ModelTrainer()
            best_model, best_score = trainer.train_and_evaluate(data_path)
            return jsonify({
                'status': 'success',
                'best_model': best_model,
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
