from sklearn.pipeline import Pipeline
from preprocessing.cleaning import DataCleaner
from features.engineering import FeatureEngineer
from preprocessing.components import build_preprocessor
import joblib
import os


def build_pipeline():
    """
    Builds the full preprocessing pipeline.
    """
    preprocessor = build_preprocessor()

    # Full Pipeline
    # DataCleaner -> FeatureEngineer -> Preprocessor
    pipeline = Pipeline(steps=[
        ('cleaner', DataCleaner()),
        ('engineer', FeatureEngineer()),
        ('preprocessor', preprocessor)
    ])

    return pipeline


def save_pipeline(pipeline, filepath='models/preprocessor.joblib'):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(pipeline, filepath)
    print(f"Pipeline saved to {filepath}")
