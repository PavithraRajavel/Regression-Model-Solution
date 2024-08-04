from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from src.utils import setup_logging


def train_linear_regression(x_train, y_train):
    logger = setup_logging()
    try:
        model = LinearRegression()
        model.fit(x_train, y_train)
        return model
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def train_decision_tree(x_train, y_train, max_depth=3, max_features=10):
    logger = setup_logging()
    try:
        model = DecisionTreeRegressor(max_depth=max_depth, max_features=max_features)
        model.fit(x_train, y_train)
        return model
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def train_random_forest(x_train, y_train, n_estimators=200):
    logger = setup_logging()
    try:
        model = RandomForestRegressor(n_estimators=n_estimators, criterion='absolute_error')
        model.fit(x_train, y_train)
        return model
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise
