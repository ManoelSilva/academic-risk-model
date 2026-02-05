import pandas as pd
import numpy as np
from src.features.engineering import FeatureEngineer


class TestFeatureEngineer:
    """Test suite for FeatureEngineer class."""

    def test_init(self):
        """Test that FeatureEngineer can be instantiated."""
        engineer = FeatureEngineer()
        assert engineer is not None
        assert isinstance(engineer, FeatureEngineer)

    def test_fit_returns_self(self):
        """Test that fit method returns self."""
        engineer = FeatureEngineer()
        X = pd.DataFrame({'col1': [1, 2, 3]})
        result = engineer.fit(X)
        assert result is engineer

    def test_fit_with_y(self):
        """Test that fit method works with y parameter."""
        engineer = FeatureEngineer()
        X = pd.DataFrame({'col1': [1, 2, 3]})
        y = pd.Series([0, 1, 0])
        result = engineer.fit(X, y)
        assert result is engineer

    def test_transform_creates_is_new_student_feature(self):
        """Test that transform creates IS_NEW_STUDENT feature correctly."""
        engineer = FeatureEngineer()
        X = pd.DataFrame({
            'SINALIZADOR_INGRESSANTE_2021': ['ingressante', 'sim', 'não', 'INGRESSANTE', 'Sim']
        })

        result = engineer.transform.__wrapped__(X)

        assert 'IS_NEW_STUDENT' in result.columns
        assert result['IS_NEW_STUDENT'].tolist() == [1, 1, 0, 1, 1]

    def test_transform_is_new_student_case_insensitive(self):
        """Test that IS_NEW_STUDENT feature is case insensitive."""
        X = pd.DataFrame({
            'SINALIZADOR_INGRESSANTE_2021': ['INGRESSANTE', 'SIM', 'Não', 'ingressante', 'SiM']
        })

        engineer = FeatureEngineer()
        result = engineer.transform.__wrapped__(X)

        assert result['IS_NEW_STUDENT'].tolist() == [1, 1, 0, 1, 1]

    def test_transform_is_new_student_with_whitespace(self):
        """Test that IS_NEW_STUDENT handles whitespace correctly."""
        X = pd.DataFrame({
            'SINALIZADOR_INGRESSANTE_2021': [' ingressante ', '  sim  ', 'não', 'ingressante']
        })

        engineer = FeatureEngineer()
        result = engineer.transform.__wrapped__(X)

        assert result['IS_NEW_STUDENT'].tolist() == [1, 1, 0, 1]

    def test_transform_is_new_student_with_numeric_values(self):
        """Test that IS_NEW_STUDENT handles numeric values correctly."""
        X = pd.DataFrame({
            'SINALIZADOR_INGRESSANTE_2021': [1, 0, 'ingressante', 'sim']
        })

        engineer = FeatureEngineer()
        result = engineer.transform.__wrapped__(X)

        # Numeric values should be converted to string and not match
        assert result['IS_NEW_STUDENT'].tolist() == [0, 0, 1, 1]

    def test_transform_is_new_student_with_nan(self):
        """Test that IS_NEW_STUDENT handles NaN values correctly."""
        X = pd.DataFrame({
            'SINALIZADOR_INGRESSANTE_2021': ['ingressante', np.nan, 'sim', None]
        })

        engineer = FeatureEngineer()
        result = engineer.transform.__wrapped__(X)

        # NaN and None should be converted to string and not match
        assert result['IS_NEW_STUDENT'].tolist() == [1, 0, 1, 0]

    def test_transform_drops_nome_column(self):
        """Test that transform drops NOME column."""
        X = pd.DataFrame({
            'NOME': ['João', 'Maria', 'Pedro'],
            'OTHER_COL': [1, 2, 3]
        })

        engineer = FeatureEngineer()
        result = engineer.transform.__wrapped__(X)

        assert 'NOME' not in result.columns
        assert 'OTHER_COL' in result.columns

    def test_transform_drops_instituicao_column(self):
        """Test that transform drops INSTITUICAO_ENSINO_ALUNO_2020 column."""
        X = pd.DataFrame({
            'INSTITUICAO_ENSINO_ALUNO_2020': ['Escola A', 'Escola B', 'Escola C'],
            'OTHER_COL': [1, 2, 3]
        })

        engineer = FeatureEngineer()
        result = engineer.transform.__wrapped__(X)

        assert 'INSTITUICAO_ENSINO_ALUNO_2020' not in result.columns
        assert 'OTHER_COL' in result.columns

    def test_transform_drops_both_identifier_columns(self):
        """Test that transform drops both identifier columns."""
        X = pd.DataFrame({
            'NOME': ['João', 'Maria', 'Pedro'],
            'INSTITUICAO_ENSINO_ALUNO_2020': ['Escola A', 'Escola B', 'Escola C'],
            'OTHER_COL': [1, 2, 3]
        })

        engineer = FeatureEngineer()
        result = engineer.transform.__wrapped__(X)

        assert 'NOME' not in result.columns
        assert 'INSTITUICAO_ENSINO_ALUNO_2020' not in result.columns
        assert 'OTHER_COL' in result.columns

    def test_transform_handles_missing_columns_gracefully(self):
        """Test that transform handles missing columns gracefully."""
        X = pd.DataFrame({
            'OTHER_COL': [1, 2, 3]
        })

        engineer = FeatureEngineer()
        result = engineer.transform.__wrapped__(X)

        # Should not raise an error
        assert 'OTHER_COL' in result.columns
        assert 'IS_NEW_STUDENT' not in result.columns

    def test_transform_does_not_modify_original_dataframe(self):
        """Test that transform does not modify the original dataframe."""
        X = pd.DataFrame({
            'NOME': ['João', 'Maria'],
            'SINALIZADOR_INGRESSANTE_2021': ['ingressante', 'sim']
        })
        original_columns = X.columns.tolist()

        engineer = FeatureEngineer()
        result = engineer.transform.__wrapped__(X)

        # Original dataframe should be unchanged
        assert X.columns.tolist() == original_columns
        assert 'NOME' in X.columns
        assert 'NOME' not in result.columns

    def test_transform_complete_workflow(self):
        """Test complete transform workflow with all features."""
        X = pd.DataFrame({
            'NOME': ['João', 'Maria', 'Pedro'],
            'INSTITUICAO_ENSINO_ALUNO_2020': ['Escola A', 'Escola B', 'Escola C'],
            'SINALIZADOR_INGRESSANTE_2021': ['ingressante', 'não', 'sim'],
            'GRADE': [8.5, 7.0, 9.0]
        })

        engineer = FeatureEngineer()
        result = engineer.transform.__wrapped__(X)

        # Check dropped columns
        assert 'NOME' not in result.columns
        assert 'INSTITUICAO_ENSINO_ALUNO_2020' not in result.columns

        # Check new feature
        assert 'IS_NEW_STUDENT' in result.columns
        assert result['IS_NEW_STUDENT'].tolist() == [1, 0, 1]

        # Check preserved columns
        assert 'GRADE' in result.columns
        assert result['GRADE'].tolist() == [8.5, 7.0, 9.0]

    def test_transform_with_empty_dataframe(self):
        """Test that transform works with empty dataframe."""
        X = pd.DataFrame()

        engineer = FeatureEngineer()
        result = engineer.transform.__wrapped__(X)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_transform_with_empty_columns(self):
        """Test that transform works with dataframe that has no matching columns."""
        X = pd.DataFrame({
            'COL1': [1, 2, 3],
            'COL2': ['a', 'b', 'c']
        })

        engineer = FeatureEngineer()
        result = engineer.transform.__wrapped__(X)

        assert 'COL1' in result.columns
        assert 'COL2' in result.columns
        assert 'IS_NEW_STUDENT' not in result.columns

    def test_fit_transform_workflow(self):
        """Test fit_transform workflow (sklearn compatibility)."""
        engineer = FeatureEngineer()
        X = pd.DataFrame({
            'NOME': ['João', 'Maria'],
            'SINALIZADOR_INGRESSANTE_2021': ['ingressante', 'sim']
        })

        engineer.fit(X)
        result = engineer.transform.__wrapped__(X)

        assert 'NOME' not in result.columns
        assert 'IS_NEW_STUDENT' in result.columns
