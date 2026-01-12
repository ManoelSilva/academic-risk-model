import pandas as pd
import numpy as np
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(file_path):
    """
    Standardized data loading.
    """
    try:
        logger.info(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path, delimiter=';')
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def map_nivel_robust(x):
    """
    Standardized logic for mapping Nivel Ideal to numeric.
    Used in cleaning and EDA.
    """
    x_str = str(x).upper()
    if pd.isna(x) or x_str == 'NAN': return np.nan
    
    # Handle ALFA
    if 'ALFA' in x_str: return 0
    
    # Handle "Fase X" or "Nivel X"
    match = re.search(r'(?:NIVEL|FASE|NVEL|NVEL)\s*(\d+)', x_str)
    if match:
        return int(match.group(1))
    
    return np.nan
