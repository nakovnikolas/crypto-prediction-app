import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.model_utility_functions import split_data
from src.logger_manager import LoggerManager

logger = LoggerManager(__name__).get_logger()


def evaluate_moving_average(train, test, window_size):
    """
    Evaluates a simple moving average model with a given window size.

    Parameters:
        train (pd.Series): Training set of the time series.
        test (pd.Series): Test set of the time series.
        window_size (int): Size of the moving average window.

    Returns:
        float: Mean Squared Error of the model on the test set.
        pd.Series: Predicted values.
    """
    # Create a list to store predictions
    predictions = []

    # Loop through each test point to generate predictions
    for i in range(len(test)):
        window_data = train[-window_size:]  # Get the last `window_size` data points from the training set
        prediction = window_data.mean()  # Calculate the average of the window
        predictions.append(prediction)

        # Update train data with the current test point
        train = pd.concat(
            [train, pd.Series([test.iloc[i]], index=[test.index[i]])]
        )  # Concatenate instead of append

    # Convert predictions list to a pandas Series
    predictions = pd.Series(predictions, index=test.index)
    error = np.mean((test - predictions) ** 2)  # Calculate Mean Squared Error
    return error, predictions


def compare_moving_average_windows(data, window_sizes):
    """
    Compares the performance of simple moving average models with different window sizes.

    Parameters:
        data (pd.Series): Time series data.
        window_sizes (list of int): List of window sizes to evaluate.

    Returns:
        dict: Dictionary with window sizes as keys and their MSE values as values.
        dict: Dictionary with window sizes as keys and their predictions as values.
    """
    # Split data into train and test sets
    train, test = split_data(data, train_ratio=0.8)

    mse_results = {}
    predictions_dict = {}

    for window_size in window_sizes:
        error, predictions = evaluate_moving_average(train, test, window_size)
        mse_results[window_size] = error
        predictions_dict[window_size] = predictions

    return mse_results, predictions_dict


def plot_moving_average_predictions(test, predictions_dict):
    """
    Plots the actual values and predictions from different moving average window sizes.

    Parameters:
        test (pd.Series): Actual test set values.
        predictions_dict (dict): Dictionary with window sizes as keys and predictions as values.
    """
    plt.figure(figsize=(14, 8))
    plt.plot(test.index, test, label='Actual Prices', color='black', linewidth=2.5)

    # Use distinct line styles and colors to differentiate each moving average window size
    styles = ['--', ':', '-.', (0, (3, 5, 1, 5)), (0, (5, 10))]
    colors = ['red', 'blue', 'green', 'orange', 'purple']

    for (window_size, predictions), style, color in zip(predictions_dict.items(), styles, colors):
        plt.plot(test.index, predictions, linestyle=style, color=color, label=f'Window Size: {window_size}', alpha=0.8)

    plt.title('Actual vs. Predicted Prices with Different Moving Average Window Sizes')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend(loc='upper left', fontsize='medium')
    plt.grid(True)
    plt.show()


def extract_ma(df: pd.DataFrame, window_sizes: list[int]) -> pd.DataFrame:
    if "price" in df.columns:
        for window in window_sizes:
            df[f"ma_{window}"] = df["price"].rolling(window).mean()
    else:
        logger.error("'Price' column not in DataFrame.")
    return df.dropna()
