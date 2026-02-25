import logging
import json
import os
from datetime import datetime
from pythonjsonlogger import jsonlogger

def setup_logger(name=__name__, log_dir='logs'):
    """
    Sets up a structured JSON logger.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Avoid adding multiple handlers if already present
    if not logger.handlers:
        # File Handler - writes to daily log file
        today = datetime.now().strftime('%Y-%m-%d')
        file_handler = logging.FileHandler(f"{log_dir}/app_{today}.log")
        
        # JSON Formatter
        formatter = jsonlogger.JsonFormatter(
            '%(asctime)s %(name)s %(levelname)s %(message)s'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Console Handler (optional, for simple viewing)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(console_handler)

    return logger
