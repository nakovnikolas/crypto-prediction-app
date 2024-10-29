import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error

from src.logger_manager import LoggerManager

logger = LoggerManager(__name__).get_logger()


def preprocess_data(df, date_col="date"):
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)
    df = df.rename(columns={"close": "price"})["price"].asfreq("D")
    return df


def mean_squared_error(y_true, y_pred):
    """
    Computes the Mean Squared Error (MSE) between true and predicted values.

    Parameters:
        y_true (array-like): Actual values from the test set.
        y_pred (array-like): Predicted values from the model.

    Returns:
        float: Mean Squared Error.
    """
    # Ensure inputs are numpy arrays for easier computation
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Compute the squared differences and then the mean
    mse = np.mean((y_true - y_pred) ** 2)
    return mse


def split_data(data, train_ratio=0.8):
    """
    Splits the time series data into training and testing sets based on the specified ratio.

    Parameters:
        data (pd.Series): The complete time series data.
        train_ratio (float): The proportion of data to be used for training (default is 0.8).

    Returns:
        pd.Series: Training data.
        pd.Series: Testing data.
    """
    # Calculate the index for splitting the data
    train_size = int(len(data) * train_ratio)

    # Split the data into train and test sets
    train = data[:train_size]
    test = data[train_size:]

    return train, test


def evaluate_model(y_test, y_pred, model_name):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    logger.info(f'{model_name} - Mean Absolute Error (MAE): {mae}')
    logger.info(f'{model_name} - Mean Squared Error (MSE): {mse}')
    return mae, mse


def cross_validation_scores(model, X, y):
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        model,
        X,
        y,
        cv=kf,
        scoring='neg_mean_absolute_error'
    )
    logger.info(f"Cross-validation MAE: {np.mean(-cv_scores):.4f} ± {np.std(-cv_scores):.4f}")
    return cv_scores


def prepare_prophet_data(df):
    prophet_df = df[['price']].rename(columns={'price': 'y'})
    prophet_df['ds'] = df.index[:len(prophet_df)]
    prophet_df = prophet_df[['ds', 'y']] 
    return prophet_df
