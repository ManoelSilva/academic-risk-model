from sklearn.metrics import classification_report, roc_auc_score, recall_score, f1_score
import mlflow


class ModelEvaluator:
    """
    Standardized evaluation module for the Academic Risk Model.
    Handles metric calculation and reporting.
    """

    @staticmethod
    def evaluate(model, X_test, y_test, threshold=0.5):
        """
        Evaluates the model on test data and returns a dictionary of metrics.
        
        Args:
            model: Trained scikit-learn model/pipeline.
            X_test: Test features.
            y_test: True labels.
            threshold (float): Decision threshold for binary classification.
            
        Returns:
            dict: Dictionary containing relevant metrics.
        """
        # Predictions
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        y_pred = (y_proba >= threshold).astype(int) if y_proba is not None else model.predict(X_test)

        # Core Metrics
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else 0.0

        metrics = {
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc
        }

        # Detailed Classification Report
        report = classification_report(y_test, y_pred, output_dict=True)
        metrics["classification_report"] = report

        return metrics

    @staticmethod
    def log_metrics_to_mlflow(metrics, step=None):
        """
        Logs a dictionary of metrics to the active MLflow run.
        """
        for k, v in metrics.items():
            if k == "classification_report":
                mlflow.log_dict(v, "classification_report.json")
            else:
                mlflow.log_metric(k, v, step=step)
