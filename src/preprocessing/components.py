from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def get_feature_lists():
    """
    Returns the lists of numeric and categorical features used in the model.
    """
    numeric_features = [
        'INDE_2021', 'IAA_2021', 'IEG_2021', 'IPS_2021', 'IDA_2021',
        'IPP_2021', 'IPV_2021', 'IAN_2021', 'DEFASAGEM_2021',
        'IDADE_ALUNO_2020', 'ANOS_PM_2020'
    ]

    categorical_features = ['PEDRA_2021', 'PONTO_VIRADA_2021']

    return numeric_features, categorical_features


def build_preprocessor():
    """
    Builds the Scikit-Learn ColumnTransformer for feature preprocessing.
    Returns:
        preprocessor (ColumnTransformer): The configured preprocessor.
    """
    numeric_features, categorical_features = get_feature_lists()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='MISSING')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )

    return preprocessor
