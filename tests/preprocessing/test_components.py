import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

from src.preprocessing.components import get_feature_lists, build_preprocessor


class TestGetFeatureLists:
    """Test suite for get_feature_lists()."""

    def test_returns_two_sequences(self):
        """Test that get_feature_lists returns exactly two sequences."""
        numeric, categorical = get_feature_lists()
        assert numeric is not None
        assert categorical is not None
        assert isinstance(numeric, (list, tuple))
        assert isinstance(categorical, (list, tuple))

    def test_numeric_features_count(self):
        """Test that numeric feature list has expected length and known names."""
        numeric, _ = get_feature_lists()
        expected = [
            'INDE', 'IAA', 'IEG', 'IPS', 'IDA',
            'IPP', 'IPV', 'IAN', 'DEFASAGEM',
            'IDADE_ALUNO', 'ANOS_PM'
        ]
        assert len(numeric) == len(expected)
        assert set(numeric) == set(expected)

    def test_categorical_features(self):
        """Test that categorical feature list is as expected."""
        _, categorical = get_feature_lists()
        expected = ['PEDRA', 'PONTO_VIRADA', 'IS_NEW_STUDENT']
        assert len(categorical) == len(expected)
        assert set(categorical) == set(expected)

    def test_no_overlap_between_numeric_and_categorical(self):
        """Test that numeric and categorical feature lists do not overlap."""
        numeric, categorical = get_feature_lists()
        assert set(numeric) & set(categorical) == set()


class TestBuildPreprocessor:
    """Test suite for build_preprocessor()."""

    def test_returns_column_transformer(self):
        """Test that build_preprocessor returns a ColumnTransformer."""
        preprocessor = build_preprocessor()
        assert preprocessor is not None
        assert isinstance(preprocessor, ColumnTransformer)

    def test_has_numeric_and_categorical_transformers(self):
        """Test that preprocessor has 'num' and 'cat' transformer names."""
        preprocessor = build_preprocessor()
        names = [name for name, *_ in preprocessor.transformers]
        assert 'num' in names
        assert 'cat' in names

    def test_transformers_use_expected_columns(self):
        """Test that transformers are configured for get_feature_lists columns."""
        numeric_features, categorical_features = get_feature_lists()
        preprocessor = build_preprocessor()
        # ColumnTransformer stores transformers as (name, transformer, columns)
        col_by_name = {name: cols for name, _, cols in preprocessor.transformers}
        assert set(col_by_name['num']) == set(numeric_features)
        assert set(col_by_name['cat']) == set(categorical_features)

    def test_remainder_is_drop(self):
        """Test that remainder columns are dropped."""
        preprocessor = build_preprocessor()
        assert preprocessor.remainder == 'drop'

    def test_fit_transform_on_numeric_and_categorical_data(self):
        """Test that preprocessor fits and transforms a dataframe with expected columns."""
        numeric_features, categorical_features = get_feature_lists()
        n_rows = 5
        X = pd.DataFrame(
            {c: np.random.randn(n_rows) for c in numeric_features},
            columns=numeric_features
        )
        for c in categorical_features:
            X[c] = ['A', 'B', 'A', 'B', 'A'][:n_rows]
        preprocessor = build_preprocessor()
        out = preprocessor.fit_transform(X)
        assert out is not None
        assert out.shape[0] == n_rows
        assert out.shape[1] >= 1  # numeric cols + one-hot encoded cat

    def test_handles_missing_numeric_values(self):
        """Test that preprocessor imputes missing numeric values."""
        numeric_features, categorical_features = get_feature_lists()
        X = pd.DataFrame(
            {c: [1.0, np.nan, 3.0] for c in numeric_features},
            columns=numeric_features
        )
        for c in categorical_features:
            X[c] = ['X', 'Y', 'X']
        preprocessor = build_preprocessor()
        out = preprocessor.fit_transform(X)
        assert not np.any(np.isnan(out))

    def test_handles_missing_categorical_values(self):
        """Test that preprocessor imputes missing categorical values."""
        numeric_features, categorical_features = get_feature_lists()
        # Categorical features: PEDRA, PONTO_VIRADA, IS_NEW_STUDENT

        X = pd.DataFrame(
            {c: [1.0, 2.0, 3.0] for c in numeric_features},
            columns=numeric_features
        )

        # Manually set columns to ensure all 3 are present
        # We need to make sure we don't assume only 2 columns if list grew
        for i, c in enumerate(categorical_features):
            if i == 0:
                X[c] = ['A', np.nan, 'A']
            elif i == 1:
                X[c] = ['P', 'P', 'Q']
            else:
                X[c] = ['X', 'Y', 'Z']  # Fallback for new columns like IS_NEW_STUDENT

        preprocessor = build_preprocessor()
        out = preprocessor.fit_transform(X)
        assert out is not None
        assert out.shape[0] == 3

    def test_output_is_numpy_array(self):
        """Test that fit_transform returns a numpy array."""
        numeric_features, categorical_features = get_feature_lists()
        X = pd.DataFrame(
            {c: [1.0, 2.0] for c in numeric_features},
            columns=numeric_features
        )
        for c in categorical_features:
            X[c] = ['A', 'B']
        preprocessor = build_preprocessor()
        out = preprocessor.fit_transform(X)
        assert isinstance(out, np.ndarray)
