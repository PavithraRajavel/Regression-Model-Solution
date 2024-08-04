from src.utils import setup_logging


def make_prediction(model, new_data):
    """
    Make predictions using the trained model.
    """
    logger = setup_logging()
    try:
        predictions = model.predict(new_data)
        return predictions
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise
