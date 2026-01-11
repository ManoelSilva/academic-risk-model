from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.preprocessing.cleaning import DataCleaner
from src.features.engineering import FeatureEngineer
import joblib
import os


def build_pipeline():
    """
    Builds the full preprocessing pipeline.
    """

    # 1. Numeric Features (e.g., INDE_2021, IAA_2021, IDADE_ALUNO_2020)
    # We select features that exist in 2021 (Last Year)
    numeric_features = [
        'INDE_2021', 'IAA_2021', 'IEG_2021', 'IPS_2021', 'IDA_2021',
        'IPP_2021', 'IPV_2021', 'IAN_2021', 'DEFASAGEM_2021',
        'IDADE_ALUNO_2020', 'ANOS_PM_2020'
    ]

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # 2. Categorical Features (e.g., PEDRA_2021, PONTO_VIRADA_2021)
    # Let's check valid columns. For now, use robust list.
    categorical_features = ['PEDRA_2021', 'PONTO_VIRADA_2021']

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='MISSING')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # 3. Column Transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'  # Drop columns not specified (like raw identifiers)
    )

    # 4. Full Pipeline
    # DataCleaner -> FeatureEngineer -> Preprocessor
    pipeline = Pipeline(steps=[
        ('cleaner', DataCleaner()),  # Custom Step 1
        ('engineer', FeatureEngineer()),  # Custom Step 2
        ('preprocessor', preprocessor)  # Scikit-Learn Step 3
    ])

    return pipeline


def save_pipeline(pipeline, filepath='models/preprocessor.joblib'):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(pipeline, filepath)
    print(f"Pipeline saved to {filepath}")
