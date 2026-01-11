import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for feature engineering:
    1. Standardizes column names (removes '_2020', '_2021' suffixes to make them generic if needed, 
       but here we will keep 2021 as the primary 'Last Year' feature set).
    2. Creates interaction features (e.g. Ratio of INDE/IAN).
    3. Handles text-based categorical columns.
    """
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        
        # Rename 2021 columns to be our "Recent" features for consistency
        # This makes the model 'year-agnostic' if we wanted to train on 2020->2021 data later.
        # For now, we just ensure we use 2021 data as the main predictors.
        
        # Example Feature: Is the student new?
        # SINALIZADOR_INGRESSANTE_2021 might be useful
        if 'SINALIZADOR_INGRESSANTE_2021' in df.columns:
            df['IS_NEW_STUDENT'] = df['SINALIZADOR_INGRESSANTE_2021'].apply(
                lambda x: 1 if str(x).strip().lower() in ['ingressante', 'sim'] else 0
            )
        
        # Impute critical missing numeric values with 0 or Median (will be handled by SimpleImputer later, 
        # but we can do domain-specific imputation here)
        
        # Drop identifiers
        cols_to_drop = ['NOME', 'INSTITUICAO_ENSINO_ALUNO_2020'] 
        df = df.drop(columns=cols_to_drop, errors='ignore')
        
        return df
