import time
import json
import shutil
import os
import logging
import re
from datetime import datetime

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline

from evaluation.evaluator import ModelEvaluator
from preprocessing.cleaning import DataCleaner
from preprocessing.components import build_preprocessor, get_feature_lists
from features.engineering import FeatureEngineer
from monitoring.metrics import TRAINING_DURATION, TRAINING_CV_SCORE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, config=None):
        """
        Initialize the ModelTrainer with a configuration dictionary.
        
        Args:
            config (dict, optional): Configuration for training. Defaults to standard values.
        """
        self.config = config or {
            "test_size": 0.2,
            "random_state": 42,
            "cv_folds": 5,
            "scoring": "recall",
            "class_weight": "balanced",
            "models_to_run": ["Logistic_Regression", "Random_Forest", "Gradient_Boosting"]
        }

        # Generate Experiment Name based on config and features
        num_feats, cat_feats = get_feature_lists()
        n_features = len(num_feats) + len(cat_feats)

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        self.experiment_name = self._build_experiment_name(n_features)

        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
        mlflow.set_experiment(self.experiment_name)
        logger.info(f"Initialized Experiment: {self.experiment_name}")
        logger.info(f"Configuration: {self.config}")

    @staticmethod
    def _sanitize_experiment_fragment(value):
        """
        Convert config values into safe fragments for MLflow experiment names.
        """
        text = str(value)
        text = text.replace(" ", "_")
        return re.sub(r"[^A-Za-z0-9._-]", "-", text)

    def _build_experiment_name(self, n_features):
        """
        Build experiment name using selected training parameter names and values.
        """
        tracked_keys = [
            "scoring",
            "cv_folds",
            "test_size",
            "random_state",
            "class_weight",
            "models_to_run",
        ]

        param_fragments = []
        for key in tracked_keys:
            raw_value = self.config.get(key)
            if isinstance(raw_value, list):
                value = "+".join([self._sanitize_experiment_fragment(v) for v in raw_value]) or "none"
            else:
                value = self._sanitize_experiment_fragment(raw_value)
            param_fragments.append(f"{key}-{value}")

        params_part = "_".join(param_fragments)
        return f"Exp_{self.timestamp}_Feats{n_features}_{params_part}"

    def save_artifacts(self, model, model_name, score, metrics=None):
        """
        Saves the model and metadata in a structured way.
        
        Structure:
        models/
          artifacts/
            {experiment_name}/
              model.joblib
              metadata.json
          production/
            model.joblib  (Copy of best model)
            metadata.json
        """
        # Define paths
        artifact_dir = os.path.join("models", "artifacts", self.experiment_name)
        os.makedirs(artifact_dir, exist_ok=True)

        model_path = os.path.join(artifact_dir, "model.joblib")
        metadata_path = os.path.join(artifact_dir, "metadata.json")

        # Save Model
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")

        # Prepare Metadata
        metadata = {
            "experiment_name": self.experiment_name,
            "timestamp": self.timestamp,
            "best_model_name": model_name,
            "best_score": score,
            "config": self.config,
            "metrics": metrics or {}
        }

        # Save Metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"Metadata saved to {metadata_path}")

        # Promote to Production (Copy)
        prod_dir = os.path.join("models", "production")
        os.makedirs(prod_dir, exist_ok=True)

        prod_model_path = os.path.join(prod_dir, "model.joblib")
        prod_metadata_path = os.path.join(prod_dir, "metadata.json")

        shutil.copy2(model_path, prod_model_path)
        shutil.copy2(metadata_path, prod_metadata_path)

        logger.info(f"Model promoted to production: {prod_model_path}")

    def prepare_data(self, data_path):
        """
        Loads data, cleans it (calculates target), and splits into X and y.
        Returns X_train, X_test, y_train, y_test.
        """
        logger.info(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path, delimiter=';')

        # Step 1: Cleaning & Target Calculation
        # We use DataCleaner explicitly to get the dataframe with TARGET
        cleaner = DataCleaner()
        df_clean = cleaner.transform(df)

        # Step 2: Feature Engineering
        target_col = 'TARGET'
        if target_col not in df_clean.columns:
            raise ValueError("Target column not found after cleaning.")

        y = df_clean[target_col]
        X = df_clean.drop(columns=[target_col])

        # Step 3: Split
        # Using stratify because of potential imbalance
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config['test_size'],
            random_state=self.config['random_state'],
            stratify=y
        )

        logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        return X_train, X_test, y_train, y_test

    def train_and_evaluate(self, data_path):
        X_train, X_test, y_train, y_test = self.prepare_data(data_path)

        # Define models to test
        all_models = {
            "Logistic_Regression": LogisticRegression(
                class_weight=self.config['class_weight'],
                max_iter=1000,
                random_state=self.config['random_state']
            ),
            "Random_Forest": RandomForestClassifier(
                class_weight=self.config['class_weight'],
                n_estimators=100,
                random_state=self.config['random_state']
            ),
            "Gradient_Boosting": GradientBoostingClassifier(
                n_estimators=100,
                random_state=self.config['random_state']
            )
        }

        # Filter models based on config
        models = {k: v for k, v in all_models.items() if k in self.config['models_to_run']}

        best_model = None
        best_score = -1
        best_model_name = ""

        # Get the preprocessing part of the pipeline
        preprocessor = build_preprocessor()

        for name, model in models.items():
            with mlflow.start_run(run_name=f"Train_{name}"):
                train_start = time.perf_counter()
                logger.info(f"Training {name}...")

                mlflow.log_params(self.config)

                full_pipeline = Pipeline(steps=[
                    ('engineer', FeatureEngineer()),
                    ('preprocessor', preprocessor),
                    ('classifier', model)
                ])

                scoring_metric = self.config['scoring']
                if scoring_metric == 'recall':
                    scorer = 'recall'
                elif scoring_metric == 'roc_auc':
                    scorer = 'roc_auc'
                else:
                    scorer = scoring_metric

                cv_scores = cross_val_score(
                    full_pipeline, X_train, y_train,
                    cv=self.config['cv_folds'],
                    scoring=scorer
                )

                mean_cv_score = np.mean(cv_scores)
                std_cv_score = np.std(cv_scores)

                logger.info(f"{name} CV {scoring_metric}: {mean_cv_score:.4f} (+/- {std_cv_score:.4f})")

                mlflow.log_metric(f"cv_mean_{scoring_metric}", mean_cv_score)
                mlflow.log_metric(f"cv_std_{scoring_metric}", std_cv_score)
                mlflow.log_param("model_type", name)

                TRAINING_CV_SCORE.labels(
                    stat="mean", metric_name=scoring_metric, model_type=name
                ).set(mean_cv_score)
                TRAINING_CV_SCORE.labels(
                    stat="std", metric_name=scoring_metric, model_type=name
                ).set(std_cv_score)

                full_pipeline.fit(X_train, y_train)

                metrics = ModelEvaluator.evaluate(full_pipeline, X_test, y_test)

                logger.info(f"Test Recall: {metrics['recall']:.4f}")
                logger.info(f"Test ROC-AUC: {metrics['roc_auc']:.4f}")

                ModelEvaluator.log_metrics_to_mlflow(metrics)

                test_recall = metrics['recall']
                test_roc_auc = metrics['roc_auc']

                mlflow.sklearn.log_model(full_pipeline, "model")

                train_elapsed = time.perf_counter() - train_start
                TRAINING_DURATION.labels(model_type=name).observe(train_elapsed)
                mlflow.log_metric("training_duration_seconds", train_elapsed)

                current_score = 0
                if scoring_metric == 'recall':
                    current_score = test_recall
                elif scoring_metric == 'roc_auc':
                    current_score = test_roc_auc
                else:
                    current_score = test_recall

                if current_score > best_score:
                    best_score = current_score
                    best_model = full_pipeline
                    best_model_name = name

        logger.info(f"\n--- Best Model Selected: {best_model_name} (Score: {best_score:.4f}) ---")

        final_metrics = {
            "score": best_score,
            "metric_type": self.config['scoring']
        }

        self.save_artifacts(best_model, best_model_name, best_score, metrics=final_metrics)
        self._register_model(best_model, best_model_name, best_score, final_metrics)

        return best_model_name, best_score

    def _register_model(self, model, model_name, score, metrics):
        """Register the best model in MLflow Model Registry with proper versioning."""
        try:
            tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
            if tracking_uri.startswith("file:"):
                logger.info("MLflow Model Registry requires a tracking server; skipping registration for local file store.")
                return

            registered_name = "academic-risk-classifier"
            with mlflow.start_run(run_name=f"Register_{model_name}"):
                mlflow.log_params({
                    "model_type": model_name,
                    "scoring": self.config["scoring"],
                    "experiment_name": self.experiment_name,
                })
                mlflow.log_metric("best_score", score)
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(k, v)

                model_info = mlflow.sklearn.log_model(
                    model,
                    artifact_path="model",
                    registered_model_name=registered_name,
                )
                logger.info(f"Model registered as '{registered_name}' version: {model_info.registered_model_version}")

        except Exception as e:
            logger.warning(f"MLflow Model Registry registration skipped: {e}")
