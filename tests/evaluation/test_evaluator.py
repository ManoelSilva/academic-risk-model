import pytest
import numpy as np
from unittest.mock import MagicMock
from src.evaluation.evaluator import ModelEvaluator


class TestModelEvaluator:
    """Test suite for ModelEvaluator class."""

    def test_evaluate_returns_correct_structure(self):
        """Test that evaluate returns expected metrics keys."""
        # Mock model
        model = MagicMock()
        model.predict.return_value = np.array([0, 1, 0, 1])
        model.predict_proba.return_value = np.array([
            [0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.3, 0.7]
        ])

        X_test = np.random.rand(4, 5)
        y_test = np.array([0, 1, 0, 1])

        metrics = ModelEvaluator.evaluate(model, X_test, y_test)

        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'roc_auc' in metrics
        assert 'classification_report' in metrics
        assert isinstance(metrics['classification_report'], dict)

    def test_evaluate_handles_models_without_predict_proba(self):
        """Test that evaluate handles models that don't support predict_proba."""
        model = MagicMock()
        del model.predict_proba  # Ensure attribute error or similar if accessed

        class SimpleModel:
            def predict(self, X):
                return np.array([0, 1])

        model = SimpleModel()
        X_test = np.random.rand(2, 2)
        y_test = np.array([0, 1])

        metrics = ModelEvaluator.evaluate(model, X_test, y_test)

        assert metrics['roc_auc'] == 0.0  # Default when no proba
        assert metrics['recall'] == 1.0

    def test_log_metrics_to_mlflow(self):
        """Test that log_metrics_to_mlflow calls mlflow functions."""
        metrics = {
            "recall": 0.8,
            "roc_auc": 0.75,
            "classification_report": {"dummy": "report"}
        }

        mlflow = pytest.importorskip('mlflow')

        from unittest.mock import patch

        with patch('src.evaluation.evaluator.mlflow') as mock_mlflow:
            ModelEvaluator.log_metrics_to_mlflow(metrics)

            mock_mlflow.log_metric.assert_any_call("recall", 0.8, step=None)
            mock_mlflow.log_metric.assert_any_call("roc_auc", 0.75, step=None)
            mock_mlflow.log_dict.assert_called_once_with({"dummy": "report"}, "classification_report.json")
