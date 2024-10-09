import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg


def preprocess_data(df, date_col="date"):
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)
    df = df["close"]
    df = df.asfreq('D')
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


def plot_acf_pacf(data, lags=20):
    """
    Plots the ACF and PACF of the time series data.

    Parameters:
        data (pd.Series): Time series data.
        lags (int): Number of lags to display in the plot.
    """
    _, ax = plt.subplots(2, 1, figsize=(7, 4))

    plot_acf(data, lags=lags, ax=ax[0])
    ax[0].set_title('Autocorrelation Function (ACF)')

    plot_pacf(data, lags=lags, ax=ax[1])
    ax[1].set_title('Partial Autocorrelation Function (PACF)')

    plt.tight_layout()
    plt.show()


def evaluate_ar_model(train, test, lag_set):
    """
    Fits an AR model and evaluates its performance using the test set.

    Parameters:
        train (pd.Series): Training set of the time series.
        test (pd.Series): Test set of the time series.
        lags (tuple): Lag values to be used in the AR model.

    Returns:
        float: Mean Squared Error of the model on the test set.
        pd.Series: Predicted values.
    """
    model = AutoReg(train, lags=lag_set)
    model_fit = model.fit()
    predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
    error = mean_squared_error(test, predictions)
    return error, predictions


def compare_lag_sets(data, lag_sets):
    """
    Compares the performance of different lag sets in AR models.

    Parameters:
        data (pd.Series): Time series data.
        lag_sets (list of tuples): List of lag sets to evaluate.

    Returns:
        dict: Dictionary with lag sets as keys and their MSE values as values.
    """
    # Train-test split (e.g., using the last 20% for testing)
    train_size = int(len(data) * 0.8)
    train, test = data[:train_size], data[train_size:]

    results = {}
    for lags in lag_sets:
        error, _ = evaluate_ar_model(train, test, lags)
        results[lags] = error
        print(f"Lags: {lags}, Test MSE: {error:.4f}")

    return results


def plot_predictions(test, predictions_dict, zoom_in=False, zoom_range=50):
    """
    Plots the actual values and predictions from different lag sets with improved visualization.

    Parameters:
        test (pd.Series): Actual test set values.
        predictions_dict (dict): Dictionary with lag sets as keys and predictions as values.
        zoom_in (bool): If True, zoom in to a subset of the test data for better visualization.
        zoom_range (int): The number of data points to display when zooming in.
    """
    plt.figure(figsize=(14, 8))

    # Determine the plot range
    if zoom_in:
        test = test[-zoom_range:]
        for lags in predictions_dict:
            predictions_dict[lags] = predictions_dict[lags][-zoom_range:]

    # Plot the actual values
    plt.plot(
        test.index,
        test,
        label='Actual Prices',
        color='black',
        linewidth=2.5
    )

    # Plot predictions with different styles and colors
    styles = ['--', ':', '-.', (0, (3, 5, 1, 5)), (0, (5, 10))]
    colors = ['red', 'blue', 'green', 'orange', 'purple']

    for (lags, predictions), style, color in zip(
        predictions_dict.items(),
        styles,
        colors
    ):
        plt.plot(
            test.index,
            predictions,
            linestyle=style,
            color=color,
            label=f'Predicted (Lags: {lags})', alpha=0.8
        )

    plt.title('Actual vs. Predicted Prices with Different Lag Sets')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend(loc='down right', fontsize='medium')
    plt.grid(True)
    plt.show()


def create_lag_features(df, n_lags):
    for lag in range(1, n_lags + 1):
        df[f'lag_{lag}'] = df["close"].shift(lag)
    df = df.dropna().reset_index()
    return df


def extract_features(df):
    # Moving averages as features
    df["ma_7"] = df["close"].rolling(window=7).mean()
    df["ma_30"] = df["close"].rolling(window=30).mean()

    # RSI as a feature
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # Add time-lagged features
    df["lag_1"] = create_lag_features(df, 1)
    df["lag_3"] = create_lag_features(df, 3)
    df["lag_7"] = create_lag_features(df, 7)

    # Fill any missing values that result from the rolling calculations
    df.fillna(method="ffill", inplace=True)

    return df
