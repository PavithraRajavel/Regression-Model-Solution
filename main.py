import logging
import os
from src.data_loading import load_data
from src.data_processing import processing
from src.model_training import train_linear_regression, train_decision_tree, train_random_forest
from src.model_evaluation import evaluation
from src.model_prediction import make_prediction
from src.utils import setup_logging
from sklearn.model_selection import train_test_split


import pickle


def main():
    logger = setup_logging()
    try:
    
        data = load_data('data/final.csv')
        logger.info("Data loaded")
        
        x, y = processing(data)
        logger.info("Feature Engineering Completed")
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)
        logger.info("Dataset Splitted into test and train set")
        
        lr_model = train_linear_regression(x_train, y_train)
        logger.info(f'Linear Regression Train MAE: {evaluation(lr_model, x_train, y_train)}')
        logger.info(f'Linear Regression Test MAE: {evaluation(lr_model, x_test, y_test)}')
        logger.info("Training Completed for Linear Regression")
        
        dt_model = train_decision_tree(x_train, y_train)        
        logger.info(f'"Decision Tree Train MAE: {evaluation(dt_model, x_train, y_train)}')
        logger.info(f'"Decision Tree Test MAE: {evaluation(dt_model, x_test, y_test)}')
        logger.info("Training Completed for Decision Tree model")
        
        rf_model = train_random_forest(x_train, y_train)
        logger.info(f'"Random Forest Train MAE: {evaluation(rf_model, x_train, y_train)}')
        logger.info(f'"Random Forest Test MAE: {evaluation(rf_model, x_test, y_test)}')
        logger.info("Training Completed for Random Forest model")
        
        logger.info("Saving Random Forest model")

        model_dir = 'models'
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, 'random_forest_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(rf_model, f)
        
        logger.info("Loading Random Forest model")
        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)
        
        logger.info("Making predictions")
        sample_data = [[2012, 216, 74, 1, 1, 618, 2000, 600, 1, 0, 0, 6, 0]]
        prediction = make_prediction(loaded_model, sample_data)
        logger.info(f'Prediction: {prediction}')
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


if __name__ == "__main__":
    main()
