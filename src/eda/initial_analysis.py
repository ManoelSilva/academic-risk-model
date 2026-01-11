import pandas as pd
import mlflow
import numpy as np

# Set up MLflow
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Phase_1_EDA")


def analyze_data(file_path):
    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path, delimiter=';')
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    print(f"Shape: {df.shape}")
    print("Columns:", df.columns.tolist())

    # Log parameters
    with mlflow.start_run():
        mlflow.log_param("dataset_path", file_path)
        mlflow.log_param("num_rows", df.shape[0])
        mlflow.log_param("num_cols", df.shape[1])

        # Check for potential target variables
        potential_targets = [col for col in df.columns if
                             'DEFASAGEM' in col.upper() or 'RISCO' in col.upper() or 'NOTA' in col.upper()]
        print("\nPotential Target Columns:", potential_targets)

        # Missing values
        missing = df.isnull().sum()
        missing_percent = (missing / df.shape[0]) * 100
        print("\nMissing Values (Top 20):")
        print(missing_percent.sort_values(ascending=False).head(20))

        # Save missing values report
        missing_df = missing_percent.to_frame(name="percent_missing")
        missing_df.to_csv("missing_values_report.csv")
        mlflow.log_artifact("missing_values_report.csv")

        # Check unique values in potential targets
        for col in potential_targets:
            if col in df.columns:
                unique_vals = df[col].nunique()
                print(f"\nUnique values in {col}: {unique_vals}")
                if unique_vals < 20:
                    print(df[col].value_counts())

        # --- Target Variable Logic ---
        print("\n--- Target Variable Identification ---")

        # Mapping logic
        # Using simpler substring matching to handle encoding/naming variations
        def map_nivel(x):
            x_str = str(x).upper()
            if pd.isna(x) or x_str == 'NAN': return np.nan

            if 'ALFA' in x_str: return 0
            if 'FASE 1' in x_str or 'NIVEL 1' in x_str: return 1
            if 'FASE 2' in x_str or 'NIVEL 2' in x_str: return 2
            if 'FASE 3' in x_str or 'NIVEL 3' in x_str: return 3
            if 'FASE 4' in x_str or 'NIVEL 4' in x_str: return 4
            if 'FASE 5' in x_str or 'NIVEL 5' in x_str: return 5
            if 'FASE 6' in x_str or 'NIVEL 6' in x_str: return 6
            if 'FASE 7' in x_str or 'NIVEL 7' in x_str: return 7
            if 'FASE 8' in x_str or 'NIVEL 8' in x_str: return 8

            return np.nan

        df['NIVEL_IDEAL_2022_NUM'] = df['NIVEL_IDEAL_2022'].apply(map_nivel)

        # Calculate Defasagem 2022
        # FASE_2022 is numeric.
        df['DEFASAGEM_2022_CALC'] = df['FASE_2022'] - df['NIVEL_IDEAL_2022_NUM']

        print("\nCalculated DEFASAGEM_2022 distribution:")
        print(df['DEFASAGEM_2022_CALC'].value_counts().sort_index())

        # Define Target: Risk of Delay (Defasagem < 0)
        # Assuming < 0 means they are in a lower phase than ideal
        df['TARGET'] = (df['DEFASAGEM_2022_CALC'] < 0).astype(int)

        print("\nTarget Variable (Risk of Delay) Distribution:")
        print(df['TARGET'].value_counts())
        print(f"Class Balance: {df['TARGET'].mean():.2%}")

        # --- EDA Visuals ---
        # Correlation
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr = df[numeric_cols].corr()

        # Top correlations with Target
        print("\nTop 10 Features correlated with Target:")
        print(corr['TARGET'].sort_values(ascending=False).head(10))
        print(corr['TARGET'].sort_values(ascending=True).head(10))

        # Log artifacts
        df[['FASE_2022', 'NIVEL_IDEAL_2022', 'DEFASAGEM_2022_CALC', 'TARGET']].head(20).to_csv(
            "target_verification.csv")
        mlflow.log_artifact("target_verification.csv")


if __name__ == "__main__":
    analyze_data("data/raw/PEDE_PASSOS_DATASET_FIAP.csv")
