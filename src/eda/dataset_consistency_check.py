import pandas as pd
from utils.data_utils import map_nivel_robust, load_data


def run_consistency_check():
    """
    Performs a consistency check on the dataset, specifically verifying:
    1. Target variable derivation logic (Defasagem).
    2. Data completeness (Missing values in critical columns).
    3. Potential data leakage (identification of future columns).
    """
    print("Loading data for consistency check...")
    try:
        df = load_data('data/raw/PEDE_PASSOS_DATASET_FIAP.csv')
    except Exception:
        return

    # 1. Analyze Nivel Ideal text patterns
    print("\n--- Nivel Ideal Patterns ---")
    print("2021 Unique:", df['NIVEL_IDEAL_2021'].unique())
    print("2022 Unique:", df['NIVEL_IDEAL_2022'].unique())

    # 2. Robust Mapping (Using Shared Logic)
    df['NIVEL_IDEAL_2022_NUM'] = df['NIVEL_IDEAL_2022'].apply(map_nivel_robust)

    # Check for unmapped values
    unmapped = df[df['NIVEL_IDEAL_2022'].notna() & df['NIVEL_IDEAL_2022_NUM'].isna()]
    if not unmapped.empty:
        print("\nWARNING: Unmapped Nivel Ideal values:")
        print(unmapped['NIVEL_IDEAL_2022'].unique())
    else:
        print("\nAll Nivel Ideal values mapped successfully.")

    # 3. Calculate Defasagem
    # Ensure FASE_2022 is treated as numeric
    df['FASE_2022'] = pd.to_numeric(df['FASE_2022'], errors='coerce')
    df['DEFASAGEM_2022_CALC'] = df['FASE_2022'] - df['NIVEL_IDEAL_2022_NUM']

    print("\nComputed Defasagem 2022 Distribution:")
    print(df['DEFASAGEM_2022_CALC'].value_counts().sort_index())

    # 4. Check Dataset overlap
    # Do we have rows with NO 2022 data?
    missing_2022 = df['FASE_2022'].isna().sum()
    print(f"\nRows missing FASE_2022 (Target Ground Truth): {missing_2022} / {len(df)}")

    # 5. Check Leakage Candidates
    # Columns ending in _2022
    cols_2022 = [c for c in df.columns if '2022' in c]
    print(f"\nColumns from 2022 (Potential Leakage): {len(cols_2022)}")
    print(cols_2022)

    # 6. Check Target Balance (on valid rows)
    df_valid = df.dropna(subset=['DEFASAGEM_2022_CALC'])
    df_valid['TARGET'] = (df_valid['DEFASAGEM_2022_CALC'] < 0).astype(int)

    print("\nTarget Distribution on Valid Data (1 = Delayed):")
    print(df_valid['TARGET'].value_counts())
    print(f"Percentage at Risk: {df_valid['TARGET'].mean():.2%}")
    print(f"Valid Dataset Size: {len(df_valid)}")


if __name__ == "__main__":
    run_consistency_check()
