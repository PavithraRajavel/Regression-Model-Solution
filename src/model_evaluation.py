from sklearn.metrics import mean_absolute_error
from src.utils import setup_logging


def evaluation(model, x, y):
    logger = setup_logging()
    try:
        predictions = model.predict(x)
        mae = mean_absolute_error(y, predictions)
        return mae
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise
