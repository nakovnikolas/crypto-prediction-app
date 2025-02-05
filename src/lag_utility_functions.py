import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg

from src.model_utility_functions import mean_squared_error, split_data
from src.logger_manager import LoggerManager

logger = LoggerManager(__name__).get_logger()


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
    predictions = model_fit.predict(
        start=len(train),
        end=len(train)+len(test)-1,
        dynamic=False
    )
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
    train, test = split_data(data)

    results = {}
    for lags in lag_sets:
        error, _ = evaluate_ar_model(train, test, lags)
        results[lags] = error
        print(f"Lags: {lags}, Test MSE: {error:.4f}")

    return results


def plot_predictions(test, predictions_dict, zoom_in=False, zoom_range=50):
    """
    Plots the actual values and predictions from different lag sets
    with improved visualization.

    Parameters:
        test (pd.Series): Actual test set values.
        predictions_dict (dict): Dictionary with lag sets as keys
        and predictions as values.
        zoom_in (bool): If True, zoom in to a subset of the test data
        for better visualization.
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
    plt.legend(loc='lower right', fontsize='medium')
    plt.grid(True)
    plt.show()


def extract_lags(df: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    """
    Add lag features to the DataFrame for the specified lags.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing a 'price' column.
        lags (list[int]): A list of integers
                          representing the lags to calculate.

    Returns:
        pd.DataFrame: The DataFrame with lag features added.
    """
    if "price" not in df.columns:
        logger.error("'Price' column not in DataFrame.")
        return df

    try:
        # Ensure the 'price' column is numeric
        df["price"] = pd.to_numeric(df["price"], errors="coerce")

        # Create lag features
        for lag in lags:
            df[f"lag_{lag}"] = df["price"].shift(lag)

        # Drop rows with NaNs introduced by shifting
        df = df.dropna()

    except Exception as e:
        logger.error(f"Error while extracting lags: {e}")
        raise

    return df
