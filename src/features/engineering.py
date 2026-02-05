from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for feature engineering.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    @staticmethod
    def transform(X):
        df = X.copy()

        # Example Feature: Is the student new?
        if 'SINALIZADOR_INGRESSANTE_2021' in df.columns:
            df['IS_NEW_STUDENT'] = df['SINALIZADOR_INGRESSANTE_2021'].apply(
                lambda x: 1 if str(x).strip().lower() in ['ingressante', 'sim'] else 0
            )

        # Drop identifiers
        cols_to_drop = ['NOME', 'INSTITUICAO_ENSINO_ALUNO_2020']
        df = df.drop(columns=cols_to_drop, errors='ignore')

        return df
