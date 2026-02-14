import pandas as pd
import numpy as np
from unittest.mock import patch

from src.preprocessing.cleaning import DataCleaner


class TestDataCleaner:
    """Test suite for DataCleaner class."""

    def test_init(self):
        """Test that DataCleaner can be instantiated."""
        cleaner = DataCleaner()
        assert cleaner is not None
        assert isinstance(cleaner, DataCleaner)
        assert cleaner.leakage_year == '2022'

    def test_fit_returns_self(self):
        """Test that fit method returns self."""
        cleaner = DataCleaner()
        X = pd.DataFrame({
            'NIVEL_IDEAL_2022': ['ALFA', 'Nivel 1'],
            'FASE_2022': [0, 1]
        })
        result = cleaner.fit(X)
        assert result is cleaner

    def test_fit_with_y(self):
        """Test that fit method works with y parameter."""
        cleaner = DataCleaner()
        X = pd.DataFrame({
            'NIVEL_IDEAL_2022': ['ALFA', 'Nivel 1'],
            'FASE_2022': [0, 1]
        })
        y = pd.Series([0, 1])
        result = cleaner.fit(X, y)
        assert result is cleaner

    @patch('builtins.print')
    def test_transform_maps_nivel_ideal_and_creates_target(self, mock_print):
        """Test that transform maps NIVEL_IDEAL_2022 to numeric and creates TARGET."""
        cleaner = DataCleaner()
        # ALFA -> 0, Nivel 1 -> 1. FASE_2022 0,1. Defasagem: 0-0=0 (not delayed), 1-1=0 (not delayed)
        X = pd.DataFrame({
            'NIVEL_IDEAL_2022': ['ALFA', 'Nivel 1'],
            'FASE_2022': [0, 1],
            'OTHER_COL': [10, 20]
        })
        result = cleaner.transform(X)
        assert 'TARGET' in result.columns
        assert result['TARGET'].tolist() == [0, 0]
        assert 'OTHER_COL' in result.columns

    @patch('builtins.print')
    def test_transform_target_one_when_defasagem_negative(self, mock_print):
        """Test that TARGET is 1 when FASE_2022 < NIVEL_IDEAL (delayed)."""
        cleaner = DataCleaner()
        # FASE 0 vs NIVEL 1 -> defasagem -1 -> TARGET 1
        X = pd.DataFrame({
            'NIVEL_IDEAL_2022': ['Nivel 1', 'Fase 2'],
            'FASE_2022': [0, 1]
        })
        result = cleaner.transform(X)
        assert result['TARGET'].tolist() == [1, 1]

    @patch('builtins.print')
    def test_transform_drops_rows_with_missing_ground_truth(self, mock_print):
        """Test that transform drops rows with missing FASE_2022 or NIVEL_IDEAL_2022_NUM."""
        cleaner = DataCleaner()
        X = pd.DataFrame({
            'NIVEL_IDEAL_2022': ['ALFA', 'Nivel 1', 'Nivel 2'],
            'FASE_2022': [0, np.nan, 2]
        })
        result = cleaner.transform(X)
        assert len(result) == 2

    @patch('builtins.print')
    def test_transform_drops_leakage_columns(self, mock_print):
        """Test that transform drops 2022 columns except TARGET and intermediate cols."""
        cleaner = DataCleaner()
        X = pd.DataFrame({
            'NIVEL_IDEAL_2022': ['ALFA', 'Nivel 1'],
            'FASE_2022': [0, 1],
            'SOME_2022_COL': [1, 2],
            'PRE_2021_COL': [3, 4]
        })
        result = cleaner.transform(X)
        assert 'TARGET' in result.columns
        assert 'NIVEL_IDEAL_2022' not in result.columns
        assert 'FASE_2022' not in result.columns
        assert 'NIVEL_IDEAL_2022_NUM' not in result.columns
        assert 'DEFASAGEM_2022_CALC' not in result.columns
        assert 'SOME_2022_COL' not in result.columns
        assert 'PRE_2021_COL' in result.columns

    @patch('builtins.print')
    def test_transform_ensures_fase_numeric(self, mock_print):
        """Test that FASE_2022 is coerced to numeric (non-numeric become NaN and rows dropped)."""
        cleaner = DataCleaner()
        X = pd.DataFrame({
            'NIVEL_IDEAL_2022': ['ALFA', 'Nivel 1', 'Nivel 1'],
            'FASE_2022': ['0', '1', 'invalid']
        })
        result = cleaner.transform(X)
        assert len(result) == 2

    @patch('builtins.print')
    def test_transform_does_not_modify_original_dataframe(self, mock_print):
        """Test that transform does not modify the original dataframe."""
        cleaner = DataCleaner()
        X = pd.DataFrame({
            'NIVEL_IDEAL_2022': ['ALFA', 'Nivel 1'],
            'FASE_2022': [0, 1]
        })
        original_columns = X.columns.tolist()
        result = cleaner.transform(X)
        assert X.columns.tolist() == original_columns
        assert 'TARGET' not in X.columns
        assert 'TARGET' in result.columns

    @patch('builtins.print')
    def test_transform_with_all_rows_missing_ground_truth(self, mock_print):
        """Test that transform returns empty dataframe when all rows have missing ground truth."""
        cleaner = DataCleaner()
        X = pd.DataFrame({
            'NIVEL_IDEAL_2022': [np.nan, np.nan],
            'FASE_2022': [np.nan, np.nan]
        })
        result = cleaner.transform(X)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    @patch('builtins.print')
    def test_transform_nivel_ideal_alfa_and_numeric(self, mock_print):
        """Test NIVEL_IDEAL mapping: ALFA -> 0, Nivel X -> X."""
        cleaner = DataCleaner()
        X = pd.DataFrame({
            'NIVEL_IDEAL_2022': ['ALFA', 'Nivel 2', 'Fase 3'],
            'FASE_2022': [0, 2, 3]
        })
        result = cleaner.transform(X)
        # Defasagem: 0-0=0, 2-2=0, 3-3=0 -> TARGET 0,0,0
        assert result['TARGET'].tolist() == [0, 0, 0]

    @patch('builtins.print')
    def test_transform_renames_columns_to_generic(self, mock_print):
        """Test that transform renames year-suffixed columns to generic names."""
        cleaner = DataCleaner()
        X = pd.DataFrame({
            'NIVEL_IDEAL_2022': ['ALFA', 'Nivel 1'],
            'FASE_2022': [0, 1],
            'INDE_2021': [1.0, 2.0],
            'IAA_2021': [3.0, 4.0]
        })
        result = cleaner.transform(X)
        assert 'INDE' in result.columns
        assert 'IAA' in result.columns
        assert 'INDE_2021' not in result.columns
        assert 'IAA_2021' not in result.columns

        # TARGET calculation should still work and be present
        assert 'TARGET' in result.columns

    @patch('builtins.print')
    def test_transform_fit_transform_workflow(self, mock_print):
        """Test fit then transform workflow."""
        cleaner = DataCleaner()
        X = pd.DataFrame({
            'NIVEL_IDEAL_2022': ['ALFA', 'Nivel 1'],
            'FASE_2022': [0, 1]
        })
        cleaner.fit(X)
        result = cleaner.transform(X)
        assert 'TARGET' in result.columns
        assert len(result) == 2
