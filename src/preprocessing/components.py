from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def get_feature_lists():
    """
    Returns the lists of numeric and categorical features used in the model.
    """
    numeric_features = [
        'INDE', 'IAA', 'IEG', 'IPS', 'IDA',
        'IPP', 'IPV', 'IAN', 'DEFASAGEM',
        'IDADE_ALUNO', 'ANOS_PM'
    ]

    categorical_features = ['PEDRA', 'PONTO_VIRADA', 'IS_NEW_STUDENT']

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
