import mlflow
import numpy as np
import pandas as pd

mlflow.set_tracking_uri("file:./mlruns")


def analyze_data(file_path):
    try:
        df = pd.read_csv(file_path, delimiter=';')
    except Exception as e:
        print(f"Error: {e}")
        return

    print("\n--- Additional Target Analysis ---")

    # Check PONTO_VIRADA
    ponto_virada_cols = [c for c in df.columns if 'PONTO_VIRADA' in c]
    for col in ponto_virada_cols:
        print(f"\n{col} Value Counts:")
        print(df[col].value_counts())

    # Check correlation of INDE with calculated Defasagem
    # Re-calculate Defasagem locally
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
    df['DEFASAGEM_2022_CALC'] = df['FASE_2022'] - df['NIVEL_IDEAL_2022_NUM']
    df['TARGET_DEFASAGEM'] = (df['DEFASAGEM_2022_CALC'] < 0).astype(int)

    # Check correlation with INDE
    print("\nCorrelation between TARGET_DEFASAGEM and INDE_2022:")
    print(df['TARGET_DEFASAGEM'].corr(df['INDE_2022']))

    # Check text columns for clues
    print("\nColumns:", df.columns.tolist())


if __name__ == "__main__":
    analyze_data("data/raw/PEDE_PASSOS_DATASET_FIAP.csv")
