from datetime import timedelta
import os
import pandas as pd
import pickle
import yaml
import importlib

from model_utility_functions import preprocess_data, split_data
from lag_utility_functions import extract_lags
from logger_manager import LoggerManager
from ma_utility_functions import extract_ma


class CryptoForecastModel:
    def __init__(self, config_path="config.yaml"):
        # Determine the root directory of the project
        current_file_path = os.path.abspath(__file__)
        src_dir = os.path.dirname(current_file_path)
        root_dir = os.path.dirname(src_dir)
        config_dir = os.path.join(root_dir, "config")

        # Set up paths
        self.models_dir = os.path.join(root_dir, "models")
        self.config_path = os.path.join(config_dir, config_path)

        # Load configuration from YAML
        with open(self.config_path, "r") as file:
            config = yaml.safe_load(file)

        # Set the logger
        self.logger = LoggerManager(__name__).get_logger()

        # General configuration
        self.random_state = config["general"].get("random_state", 42)

        # Model loading based on configuration
        model_library = config["model"]["library"]
        model_name = config["model"]["name"]
        model_params = config["model"].get("model_params", {})

        # Dynamically load the model class
        model_class = self._get_model_class(model_library, model_name)
        self.model = model_class(
            random_state=self.random_state,
            **model_params
        )

        # Lists of features
        self.lags = list(map(int, config["features"]["lags"].split(",")))
        self.windows = list(map(int, config["features"]["windows"].split(",")))

        # Training parameters
        self.test_size = config["training"]["test_size"]
        self.cv_folds = config["training"]["cross_validation_folds"]

        # Load model path
        self.model_load_path = os.path.join(
            self.models_dir, config["chosen_trained_model"]
        )

        # Save model path
        self.model_save_path = config["output"]["model_save_path"]

        # Get crypto currency
        self.crypto_mapping = config["currency_mapping"]

    def _get_model_class(self, library, model_type):
        """Dynamically load model class from specified library."""
        try:
            # Import the module specified in the YAML config
            model_module = importlib.import_module(library)
            return getattr(model_module, model_type)
        except (AttributeError, ModuleNotFoundError) as e:
            raise ValueError(
                f"Model {model_type} not found in {library}. Error: {e}"
            )

    def preprocess_and_combine_data(self, crypto_data_dict):
        combined_data = []
        for currency_id, (_, df) in\
                enumerate(crypto_data_dict.items()):
            # Preprocess and add features
            df = preprocess_data(df)
            df = extract_lags(df, self.lags)
            df = extract_ma(df, self.windows)

            # Add currency identifier column
            df["currency_id"] = currency_id

            # Append processed DataFrame to list
            combined_data.append(df)

        # Concatenate all cryptocurrencies' DataFrames
        combined_df = pd.concat(combined_data)

        # Create the target variable as the next day's price
        # and drop rows with NaNs
        combined_df["y"] = combined_df["price"].shift(-1)
        combined_df = combined_df.dropna()

        # Define feature and target variables
        X = combined_df.drop(columns=['price', 'y'])
        y = combined_df['y']

        return X, y

    def prepare_train_test_split(self, X, y):
        """
        Split the data into training and testing sets
        based on the test size.
        """
        X_train, X_test = split_data(X, test_size=self.test_size)
        y_train, y_test = split_data(y, test_size=self.test_size)

        return X_train, X_test, y_train, y_test

    def train_model(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        """Train the model on the training data."""
        self.model.fit(X_train, y_train)

    def forecast(self, X: pd.DataFrame, symbol: str, future_days: int = 30):
        """Generate forecasted values for the next specified number of days."""
        predictions = []
        # Get the currency id of the chosen currency
        currency_id = self.crypto_mapping[symbol]

        # Copy the data for that currency
        history = X[X["currency_id"] == currency_id].copy()

        # Ensure the 'price' column is numeric
        history["price"] = pd.to_numeric(history["price"], errors="coerce")
        history = history.dropna(subset=["price"])

        for _ in range(future_days):
            # Get the last row
            last_row = history.iloc[-1]

            # Make the prediction
            pred = self.model.predict([last_row])[0]
            predictions.append(pred)

            # Generate the row  for the next day
            tomorrow_date = last_row.name + timedelta(days=1)
            new_row = last_row.copy()
            new_row["price"] = pred
            new_row.name = tomorrow_date

            # Add the row for the next day to the history
            history = pd.concat([history, new_row.to_frame().T])

            # Recalculate the features for the whole history
            extract_lags(history, self.lags)
            extract_ma(history, self.windows)

        # Generate a DataFrame with only the forecasted values and their dates
        forecast_dates = pd.date_range(
            start=history.index[-future_days], periods=future_days, freq="D"
        )
        forecasted_prices = predictions

        forecast_df = pd.DataFrame({
            "date": forecast_dates,
            "predicted_price": forecasted_prices
        })

        return forecast_df

    def read_model(self):
        """Load the trained model from the specified path
        and assign it to self.model."""
        try:
            with open(self.model_load_path, "rb") as f:
                self.model = pickle.load(f)
            self.logger.info(f"Model loaded from {self.model_load_path}")
        except FileNotFoundError as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def save_model(self):
        """Save the trained model to the specified path."""
        with open(self.model_save_path, "wb") as f:
            pickle.dump(self.model, f)
        self.logger.info(f"Model saved to {self.model_save_path}")
