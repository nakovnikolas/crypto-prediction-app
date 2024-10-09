import numpy as np
import pandas as pd



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
