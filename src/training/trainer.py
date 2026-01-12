import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from preprocessing.cleaning import DataCleaner
from preprocessing.components import build_preprocessor, get_feature_lists
from features.engineering import FeatureEngineer
import joblib
import os
import logging
from datetime import datetime

# Configure logging
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

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")

        # Construct meaningful experiment name
        self.experiment_name = (
            f"Exp_{timestamp}_"
            f"Feats{n_features}_"
            f"CV{self.config['cv_folds']}_"
            f"{self.config['scoring'].capitalize()}"
        )

        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment(self.experiment_name)
        logger.info(f"Initialized Experiment: {self.experiment_name}")
        logger.info(f"Configuration: {self.config}")

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
                logger.info(f"Training {name}...")

                # Log hyperparameters
                mlflow.log_params(self.config)

                # Construct full pipeline: FeatureEngineer -> ColumnTransformer -> Model
                full_pipeline = Pipeline(steps=[
                    ('engineer', FeatureEngineer()),
                    ('preprocessor', preprocessor),
                    ('classifier', model)
                ])

                # Cross-Validation
                scoring_metric = self.config['scoring']
                # Ensure scorer is valid (simple check)
                if scoring_metric == 'recall':
                    scorer = 'recall'
                elif scoring_metric == 'roc_auc':
                    scorer = 'roc_auc'
                else:
                    scorer = scoring_metric  # fallback to string

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

                # Fit on full training set
                full_pipeline.fit(X_train, y_train)

                # Evaluate on Test Set
                y_pred = full_pipeline.predict(X_test)
                y_proba = full_pipeline.predict_proba(X_test)[:, 1] if hasattr(full_pipeline, "predict_proba") else None

                # Metrics
                test_recall = recall_score(y_test, y_pred)
                test_roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else 0
                test_f1 = f1_score(y_test, y_pred)

                logger.info(f"Test Recall: {test_recall:.4f}")
                logger.info(f"Test ROC-AUC: {test_roc_auc:.4f}")

                mlflow.log_metric("test_recall", test_recall)
                mlflow.log_metric("test_roc_auc", test_roc_auc)
                mlflow.log_metric("test_f1", test_f1)

                # Log Classification Report as artifact
                report = classification_report(y_test, y_pred, output_dict=True)
                mlflow.log_dict(report, "classification_report.json")

                # Log Model
                mlflow.sklearn.log_model(full_pipeline, "model")

                # Model Selection Logic (Based on configured scoring metric)
                # We default to comparing the TEST score of the chosen metric
                current_score = 0
                if scoring_metric == 'recall':
                    current_score = test_recall
                elif scoring_metric == 'roc_auc':
                    current_score = test_roc_auc
                else:
                    current_score = test_recall  # fallback

                if current_score > best_score:
                    best_score = current_score
                    best_model = full_pipeline
                    best_model_name = name

        logger.info(f"\n--- Best Model Selected: {best_model_name} (Score: {best_score:.4f}) ---")

        # Save Best Model locally
        os.makedirs("models", exist_ok=True)
        joblib.dump(best_model, "models/best_model.joblib")
        logger.info("Best model saved to models/best_model.joblib")

        return best_model_name, best_score
