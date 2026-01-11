import pandas as pd
import numpy as np
import re
from sklearn.base import BaseEstimator, TransformerMixin


class DataCleaner(BaseEstimator, TransformerMixin):
    """
    Custom transformer to clean the raw dataset:
    1. Filter rows with missing target (ground truth).
    2. Map NIVEL_IDEAL text to numeric.
    3. Calculate Target (Risk of Defasagem).
    4. Drop leakage columns (2022 data).
    """

    def __init__(self):
        self.leakage_year = '2022'

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        print(f"Original shape: {df.shape}")

        # 1. Map NIVEL_IDEAL_2022 to numeric
        df['NIVEL_IDEAL_2022_NUM'] = df['NIVEL_IDEAL_2022'].apply(self._map_nivel_robust)

        # 2. Ensure FASE_2022 is numeric
        df['FASE_2022'] = pd.to_numeric(df['FASE_2022'], errors='coerce')

        # 3. Drop rows with missing ground truth
        df = df.dropna(subset=['FASE_2022', 'NIVEL_IDEAL_2022_NUM'])
        print(f"Shape after filtering missing ground truth: {df.shape}")

        # 4. Calculate Target
        df['DEFASAGEM_2022_CALC'] = df['FASE_2022'] - df['NIVEL_IDEAL_2022_NUM']
        # Target = 1 if Defasagem < 0 (Delayed), else 0
        df['TARGET'] = (df['DEFASAGEM_2022_CALC'] < 0).astype(int)

        # 5. Drop Leakage Columns
        # Keep only the calculated TARGET, drop all other 2022 columns
        cols_to_drop = [c for c in df.columns if self.leakage_year in c and c != 'TARGET']
        # Also drop the intermediate calculation columns if they were created
        cols_to_drop.extend(['DEFASAGEM_2022_CALC', 'NIVEL_IDEAL_2022_NUM'])

        df = df.drop(columns=cols_to_drop, errors='ignore')
        print(f"Shape after dropping leakage columns: {df.shape}")

        return df

    @staticmethod
    def _map_nivel_robust(x):
        x_str = str(x).upper()
        if pd.isna(x) or x_str == 'NAN': return np.nan

        if 'ALFA' in x_str: return 0

        match = re.search(r'(?:NIVEL|FASE|NVEL|NVEL)\s*(\d+)', x_str)
        if match:
            return int(match.group(1))

        return np.nan
