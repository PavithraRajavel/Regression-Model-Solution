import pandas as pd
from src.utils import setup_logging


def load_data(filepath):
    """
    Load dataset from a CSV file.
    """
    logger = setup_logging()
    try:
        data = pd.read_csv(filepath)
        return data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise
