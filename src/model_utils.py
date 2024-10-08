import pandas as pd


def preprocess_data(df, date_col="date"):
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)
    return df


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
