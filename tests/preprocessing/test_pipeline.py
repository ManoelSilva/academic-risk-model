import os
import tempfile
from unittest.mock import patch, MagicMock

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from src.preprocessing.components import get_feature_lists, build_preprocessor
from src.preprocessing.pipeline import build_pipeline, save_pipeline
from src.preprocessing.cleaning import DataCleaner
from src.features.engineering import FeatureEngineer


class TestBuildPipeline:
    """Test suite for build_pipeline()."""

    def test_returns_pipeline(self):
        """Test that build_pipeline returns a sklearn Pipeline."""
        pipeline = build_pipeline()
        assert pipeline is not None
        assert isinstance(pipeline, Pipeline)

    def test_pipeline_has_three_steps(self):
        """Test that pipeline has exactly three steps: cleaner, engineer, preprocessor."""
        pipeline = build_pipeline()
        step_names = [name for name, _ in pipeline.steps]
        assert len(step_names) == 3
        assert step_names == ['cleaner', 'engineer', 'preprocessor']

    def test_pipeline_first_step_is_cleaner(self):
        """Test that first step is named cleaner."""
        pipeline = build_pipeline()
        assert pipeline.steps[0][0] == 'cleaner'

    def test_pipeline_second_step_is_engineer(self):
        """Test that second step is named engineer."""
        pipeline = build_pipeline()
        assert pipeline.steps[1][0] == 'engineer'

    def test_pipeline_third_step_is_preprocessor(self):
        """Test that third step is named preprocessor."""
        pipeline = build_pipeline()
        assert pipeline.steps[2][0] == 'preprocessor'

    def test_preprocessor_step_is_column_transformer(self):
        """Test that the preprocessor step is a ColumnTransformer."""
        pipeline = build_pipeline()
        preprocessor = pipeline.steps[2][1]
        assert isinstance(preprocessor, ColumnTransformer)

    @patch('src.preprocessing.pipeline.DataCleaner')
    @patch('src.preprocessing.pipeline.FeatureEngineer')
    def test_build_pipeline_uses_cleaner_and_engineer(self, mock_engineer, mock_cleaner):
        """Test that build_pipeline instantiates DataCleaner and FeatureEngineer."""
        mock_cleaner.return_value = MagicMock()
        mock_engineer.return_value = MagicMock()
        build_pipeline()
        mock_cleaner.assert_called_once()
        mock_engineer.assert_called_once()


class TestSavePipeline:
    """Test suite for save_pipeline()."""

    def test_save_pipeline_accepts_pipeline_and_filepath(self):
        """Test that save_pipeline accepts pipeline and filepath."""
        pl = build_pipeline()
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'subdir', 'pipe.joblib')
            save_pipeline(pl, filepath)
            assert os.path.isfile(filepath)

    def test_save_pipeline_creates_directory(self):
        """Test that save_pipeline creates parent directory if needed."""
        pl = build_pipeline()
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'newdir', 'nested', 'pipe.joblib')
            assert not os.path.exists(os.path.dirname(filepath))
            save_pipeline(pl, filepath)
            assert os.path.isdir(os.path.dirname(filepath))
            assert os.path.isfile(filepath)

    def test_saved_pipeline_can_be_loaded(self):
        """Test that a saved pipeline can be loaded with joblib."""
        pl = build_pipeline()
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
            filepath = f.name
        try:
            save_pipeline(pl, filepath)
            loaded = joblib.load(filepath)
            assert isinstance(loaded, Pipeline)
            assert [s[0] for s in loaded.steps] == ['cleaner', 'engineer', 'preprocessor']
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    @patch('builtins.print')
    def test_save_pipeline_prints_message(self, mock_print):
        """Test that save_pipeline prints the save path."""
        pl = build_pipeline()
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
            filepath = f.name
        try:
            save_pipeline(pl, filepath)
            mock_print.assert_called_once()
            call_args = mock_print.call_args[0][0]
            assert filepath in call_args or 'saved' in call_args.lower()
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)


class _FeatureEngineerAdapter(BaseEstimator, TransformerMixin):
    """Adapter so Pipeline can call FeatureEngineer's static transform with one argument."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        transform_fn = getattr(FeatureEngineer.transform, '__wrapped__', FeatureEngineer.transform)
        return transform_fn(X)


class TestBuildPipelineIntegration:
    """Integration tests for the full pipeline (fit_transform with minimal data)."""

    def _minimal_df_for_pipeline(self, n_rows=5, seed=42):
        """Build a minimal DataFrame that passes cleaner -> engineer -> preprocessor."""
        np.random.seed(seed)
        numeric_features, categorical_features = get_feature_lists()
        df = pd.DataFrame({
            'NIVEL_IDEAL_2022': ['ALFA', 'Nivel 1', 'Nivel 1', 'ALFA', 'Nivel 2'][:n_rows],
            'FASE_2022': [0, 1, 1, 0, 2][:n_rows],
            'NOME': ['A', 'B', 'C', 'D', 'E'][:n_rows],
            'INSTITUICAO_ENSINO_ALUNO_2020': ['I1', 'I2', 'I3', 'I4', 'I5'][:n_rows],
            'SINALIZADOR_INGRESSANTE_2021': ['ingressante', 'sim', 'não', 'sim', 'não'][:n_rows],
        })
        for c in numeric_features:
            df[c] = np.random.randn(n_rows)
        for c in categorical_features:
            df[c] = ['X', 'Y', 'X', 'Y', 'X'][:n_rows]
        return df

    def _build_integration_pipeline(self):
        """Build pipeline with src imports and adapter so transform works in Pipeline."""
        return Pipeline(steps=[
            ('cleaner', DataCleaner()),
            ('engineer', _FeatureEngineerAdapter()),
            ('preprocessor', build_preprocessor())
        ])

    @patch('builtins.print')
    def test_pipeline_fit_transform_returns_array(self, mock_print):
        """Test that pipeline fit_transform runs and returns a numpy array."""
        pipeline = self._build_integration_pipeline()
        X = self._minimal_df_for_pipeline()
        y = X.shape[0] * [0]  # dummy target for fit
        pipeline.fit(X, y)
        out = pipeline.transform(X)
        assert out is not None
        assert isinstance(out, np.ndarray)
        assert out.shape[0] == X.shape[0]

    @patch('builtins.print')
    def test_pipeline_fit_transform_output_has_no_nans(self, mock_print):
        """Test that pipeline fit_transform output has no NaNs for valid input."""
        pipeline = self._build_integration_pipeline()
        X = self._minimal_df_for_pipeline()
        pipeline.fit(X)
        out = pipeline.transform(X)
        assert not np.any(np.isnan(out))
