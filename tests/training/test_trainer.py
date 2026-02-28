from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.components import get_feature_lists
from src.training.trainer import ModelTrainer


def _minimal_cleaned_df(n_rows=20, seed=42):
    """Build a minimal DataFrame as returned after cleaning (has TARGET + feature columns)."""
    np.random.seed(seed)
    numeric_features, categorical_features = get_feature_lists()
    df = pd.DataFrame(
        {c: np.random.randn(n_rows) for c in numeric_features},
        columns=numeric_features
    )
    for c in categorical_features:
        df[c] = np.random.choice(['A', 'B'], size=n_rows)
    # TARGET: need both classes for stratify
    df['TARGET'] = np.array([0, 1] * (n_rows // 2) + [0] * (n_rows % 2))[:n_rows]
    return df


class TestModelTrainerInit:
    """Test suite for ModelTrainer.__init__."""

    def test_init_with_default_config(self):
        """Test that ModelTrainer initializes with default config when config is None."""
        with patch('src.training.trainer.mlflow.set_tracking_uri'), \
                patch('src.training.trainer.mlflow.set_experiment'):
            trainer = ModelTrainer()
        assert trainer.config is not None
        assert trainer.config['test_size'] == 0.2
        assert trainer.config['random_state'] == 42
        assert trainer.config['cv_folds'] == 5
        assert trainer.config['scoring'] == 'recall'
        assert 'Logistic_Regression' in trainer.config['models_to_run']

    def test_init_with_custom_config(self):
        """Test that ModelTrainer uses provided config."""
        custom_config = {
            'test_size': 0.3,
            'random_state': 99,
            'cv_folds': 3,
            'scoring': 'roc_auc',
            'class_weight': 'balanced',
            'models_to_run': ['Logistic_Regression']
        }
        with patch('src.training.trainer.mlflow.set_tracking_uri'), \
                patch('src.training.trainer.mlflow.set_experiment'):
            trainer = ModelTrainer(config=custom_config)
        assert trainer.config['test_size'] == 0.3
        assert trainer.config['random_state'] == 99
        assert trainer.config['cv_folds'] == 3
        assert trainer.config['scoring'] == 'roc_auc'

    def test_experiment_name_set(self):
        """Test that experiment_name is set and contains expected substrings."""
        with patch('src.training.trainer.mlflow.set_tracking_uri'), \
                patch('src.training.trainer.mlflow.set_experiment'):
            trainer = ModelTrainer()
        assert hasattr(trainer, 'experiment_name')
        assert 'Exp_' in trainer.experiment_name
        assert 'cv_folds-5' in trainer.experiment_name
        assert 'scoring-recall' in trainer.experiment_name

    def test_experiment_name_reflects_custom_params(self):
        """Test experiment name includes custom parameter names and values."""
        custom_config = {
            'test_size': 0.3,
            'random_state': 99,
            'cv_folds': 3,
            'scoring': 'roc_auc',
            'class_weight': 'balanced',
            'models_to_run': ['Logistic_Regression']
        }
        with patch('src.training.trainer.mlflow.set_tracking_uri'), \
                patch('src.training.trainer.mlflow.set_experiment'):
            trainer = ModelTrainer(config=custom_config)

        assert 'test_size-0.3' in trainer.experiment_name
        assert 'random_state-99' in trainer.experiment_name
        assert 'cv_folds-3' in trainer.experiment_name
        assert 'scoring-roc_auc' in trainer.experiment_name
        assert 'models_to_run-Logistic_Regression' in trainer.experiment_name


class TestPrepareData:
    """Test suite for ModelTrainer.prepare_data."""

    @patch('src.training.trainer.DataCleaner')
    @patch('src.training.trainer.pd.read_csv')
    def test_prepare_data_returns_four_arrays(self, mock_read_csv, mock_cleaner_class):
        """Test that prepare_data returns X_train, X_test, y_train, y_test."""
        df_clean = _minimal_cleaned_df(n_rows=20)
        mock_cleaner = MagicMock()
        mock_cleaner.transform.return_value = df_clean
        mock_cleaner_class.return_value = mock_cleaner
        mock_read_csv.return_value = df_clean  # not used for transform output

        with patch('src.training.trainer.mlflow.set_tracking_uri'), \
                patch('src.training.trainer.mlflow.set_experiment'):
            trainer = ModelTrainer()
        X_train, X_test, y_train, y_test = trainer.prepare_data('/fake/path.csv')
        assert X_train is not None
        assert X_test is not None
        assert y_train is not None
        assert y_test is not None
        assert len(X_train) + len(X_test) == 20
        assert len(y_train) + len(y_test) == 20

    @patch('src.training.trainer.DataCleaner')
    @patch('src.training.trainer.pd.read_csv')
    def test_prepare_data_raises_when_target_missing(self, mock_read_csv, mock_cleaner_class):
        """Test that prepare_data raises ValueError when TARGET is not in cleaned df."""
        df_no_target = _minimal_cleaned_df(n_rows=10).drop(columns=['TARGET'])
        mock_cleaner = MagicMock()
        mock_cleaner.transform.return_value = df_no_target
        mock_cleaner_class.return_value = mock_cleaner
        mock_read_csv.return_value = df_no_target

        with patch('src.training.trainer.mlflow.set_tracking_uri'), \
                patch('src.training.trainer.mlflow.set_experiment'):
            trainer = ModelTrainer()
        with pytest.raises(ValueError, match="Target column not found"):
            trainer.prepare_data('/fake/path.csv')

    @patch('src.training.trainer.DataCleaner')
    @patch('src.training.trainer.pd.read_csv')
    def test_prepare_data_respects_test_size(self, mock_read_csv, mock_cleaner_class):
        """Test that train/test split respects config test_size."""
        df_clean = _minimal_cleaned_df(n_rows=100)
        mock_cleaner = MagicMock()
        mock_cleaner.transform.return_value = df_clean
        mock_cleaner_class.return_value = mock_cleaner
        mock_read_csv.return_value = df_clean

        full_config = {
            'test_size': 0.25,
            'random_state': 42,
            'cv_folds': 5,
            'scoring': 'recall',
            'class_weight': 'balanced',
            'models_to_run': ['Logistic_Regression', 'Random_Forest', 'Gradient_Boosting']
        }
        with patch('src.training.trainer.mlflow.set_tracking_uri'), \
                patch('src.training.trainer.mlflow.set_experiment'):
            trainer = ModelTrainer(config=full_config)
        X_train, X_test, y_train, y_test = trainer.prepare_data('/fake/path.csv')
        assert len(X_test) == 25
        assert len(X_train) == 75


def _make_mock_pipeline(*args, **kwargs):
    """Create a mock pipeline that supports fit, predict, predict_proba, clone, and score for cross_val_score and evaluation."""
    mock = MagicMock()
    mock.fit.return_value = mock
    mock.clone.return_value = mock
    mock.score.side_effect = lambda X, y: 0.75
    mock.predict.side_effect = lambda X: np.zeros(
        X.shape[0] if hasattr(X, 'shape') else len(X), dtype=int
    )
    mock.predict_proba.side_effect = lambda X: np.column_stack([
        np.ones(X.shape[0] if hasattr(X, 'shape') else len(X)),
        np.zeros(X.shape[0] if hasattr(X, 'shape') else len(X))
    ])
    return mock


class TestTrainAndEvaluate:
    """Test suite for ModelTrainer.train_and_evaluate."""

    @patch('src.training.trainer.ModelEvaluator')
    @patch('src.training.trainer.cross_val_score')
    @patch('src.training.trainer.Pipeline')
    @patch('src.training.trainer.shutil.copy2')
    @patch('src.training.trainer.json.dump')
    @patch('builtins.open', create=True)
    @patch('src.training.trainer.os.makedirs')
    @patch('src.training.trainer.joblib.dump')
    @patch('src.training.trainer.mlflow.sklearn.log_model')
    @patch('src.training.trainer.mlflow.log_dict')
    @patch('src.training.trainer.mlflow.log_metric')
    @patch('src.training.trainer.mlflow.log_param')
    @patch('src.training.trainer.mlflow.log_params')
    @patch('src.training.trainer.mlflow.start_run')
    @patch('src.training.trainer.DataCleaner')
    @patch('src.training.trainer.pd.read_csv')
    def test_train_and_evaluate_returns_tuple(
            self,
            mock_read_csv,
            mock_cleaner_class,
            mock_start_run,
            mock_log_params,
            mock_log_param,
            mock_log_metric,
            mock_log_dict,
            mock_log_model,
            mock_joblib_dump,
            mock_makedirs,
            mock_open,
            mock_json_dump,
            mock_copy2,
            mock_pipeline_class,
            mock_cross_val_score,
            mock_evaluator
    ):
        """Test that train_and_evaluate returns (best_model_name, best_score)."""
        df_clean = _minimal_cleaned_df(n_rows=30)
        mock_cleaner = MagicMock()
        mock_cleaner.transform.return_value = df_clean
        mock_cleaner_class.return_value = mock_cleaner
        mock_read_csv.return_value = df_clean
        mock_start_run.return_value.__enter__ = MagicMock(return_value=None)
        mock_start_run.return_value.__exit__ = MagicMock(return_value=None)
        mock_open.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_open.return_value.__exit__ = MagicMock(return_value=None)

        mock_cross_val_score.return_value = np.array([0.7, 0.8])
        mock_pipeline_class.side_effect = _make_mock_pipeline
        
        # Mock Evaluator
        mock_evaluator.evaluate.return_value = {
            'recall': 0.8, 'roc_auc': 0.75, 'f1_score': 0.78, 'classification_report': {}
        }

        with patch('src.training.trainer.mlflow.set_tracking_uri'), \
                patch('src.training.trainer.mlflow.set_experiment'):
            trainer = ModelTrainer(config={
                'test_size': 0.2,
                'random_state': 42,
                'cv_folds': 2,
                'scoring': 'recall',
                'class_weight': 'balanced',
                'models_to_run': ['Logistic_Regression']
            })
        result = trainer.train_and_evaluate('/fake/path.csv')
        assert isinstance(result, tuple)
        assert len(result) == 2
        name, score = result
        assert name in ['Logistic_Regression', 'Random_Forest', 'Gradient_Boosting']
        assert isinstance(score, (int, float))

    @patch('src.training.trainer.ModelEvaluator')
    @patch('src.training.trainer.cross_val_score')
    @patch('src.training.trainer.Pipeline')
    @patch('src.training.trainer.shutil.copy2')
    @patch('src.training.trainer.json.dump')
    @patch('builtins.open', create=True)
    @patch('src.training.trainer.os.makedirs')
    @patch('src.training.trainer.joblib.dump')
    @patch('src.training.trainer.mlflow.sklearn.log_model')
    @patch('src.training.trainer.mlflow.log_dict')
    @patch('src.training.trainer.mlflow.log_metric')
    @patch('src.training.trainer.mlflow.log_param')
    @patch('src.training.trainer.mlflow.log_params')
    @patch('src.training.trainer.mlflow.start_run')
    @patch('src.training.trainer.DataCleaner')
    @patch('src.training.trainer.pd.read_csv')
    def test_train_and_evaluate_saves_artifacts(
            self,
            mock_read_csv,
            mock_cleaner_class,
            mock_start_run,
            mock_log_params,
            mock_log_param,
            mock_log_metric,
            mock_log_dict,
            mock_log_model,
            mock_joblib_dump,
            mock_makedirs,
            mock_open,
            mock_json_dump,
            mock_copy2,
            mock_pipeline_class,
            mock_cross_val_score,
            mock_evaluator
    ):
        """Test that train_and_evaluate calls save_artifacts and persists model/metadata."""
        df_clean = _minimal_cleaned_df(n_rows=30)
        mock_cleaner = MagicMock()
        mock_cleaner.transform.return_value = df_clean
        mock_cleaner_class.return_value = mock_cleaner
        mock_read_csv.return_value = df_clean
        mock_start_run.return_value.__enter__ = MagicMock(return_value=None)
        mock_start_run.return_value.__exit__ = MagicMock(return_value=None)

        mock_cross_val_score.return_value = np.array([0.7, 0.8])
        mock_pipeline_class.side_effect = _make_mock_pipeline

        # Mock Evaluator
        mock_evaluator.evaluate.return_value = {
            'recall': 0.8, 'roc_auc': 0.75, 'f1_score': 0.78, 'classification_report': {}
        }

        # Mock open context manager so save_artifacts can write metadata
        mock_open.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_open.return_value.__exit__ = MagicMock(return_value=None)

        with patch('src.training.trainer.mlflow.set_tracking_uri'), \
                patch('src.training.trainer.mlflow.set_experiment'):
            trainer = ModelTrainer(config={
                'test_size': 0.2,
                'random_state': 42,
                'cv_folds': 2,
                'scoring': 'recall',
                'class_weight': 'balanced',
                'models_to_run': ['Logistic_Regression']
            })

        trainer.train_and_evaluate('/fake/path.csv')

        # Check that joblib.dump was called (saving the artifact)
        assert mock_joblib_dump.called

        # Check that directories were created
        assert mock_makedirs.called

        # Check that metadata was saved (json.dump called)
        assert mock_json_dump.called

        # Check that model was promoted to production (shutil.copy2 called)
        assert mock_copy2.called
        assert mock_copy2.call_count >= 1
