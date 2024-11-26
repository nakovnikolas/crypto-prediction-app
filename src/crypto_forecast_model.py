from datetime import timedelta
import pandas as pd
import pickle
import yaml
import importlib

from sklearn.model_selection import cross_val_score

from model_utility_functions import preprocess_data, split_data
from lag_utility_functions import extract_lags
from logger_manager import LoggerManager
from ma_utility_functions import extract_ma


class CryptoForecastModel:
    def __init__(self, config_path="config.yaml"):
        # Load configuration from YAML
        with open(config_path, "r") as file:
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
        self.lags = config["features"]["lags"]
        self.windows = config["features"]["windows"]

        # Training parameters
        self.test_size = config["training"]["test_size"]
        self.cv_folds = config["training"]["cross_validation_folds"]

        # Save model path
        self.model_save_path = config["output"]["model_save_path"]

        # Initialize data containers
        self.data = None
        self.features = None
        self.labels = None

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

    def train_model(self, X_train, y_train):
        """Train the model on the training data with cross-validation."""
        # Cross-validation
        scores = cross_val_score(
            self.model,
            X_train,
            y_train,
            cv=self.cv_folds,
            scoring="neg_mean_squared_error"
        )
        self.model.fit(X_train, y_train)

        self.logger.info(f"Cross-Validation MSE: {-scores.mean()}")

    def forecast(self, X_last, future_days=30):
        """Generate forecasted values for the next specified number of days."""
        predictions = []
        last_row = X_last.iloc[-1]

        for _ in range(future_days):
            pred = self.model.predict([last_row])[0]
            predictions.append(pred)

            # Shift features for next day prediction
            last_row = last_row.shift(1)
            last_row.iloc[0] = pred

        # Return predictions as a DataFrame with dates
        forecast_dates = pd.date_range(
            start=X_last.index[-1] + timedelta(days=1),
            periods=future_days
        )
        forecast_df = pd.DataFrame(
            {"date": forecast_dates, "forecasted_price": predictions}
        )
        return forecast_df.set_index("date")

    def save_model(self):
        """Save the trained model to the specified path."""
        with open(self.model_save_path, "wb") as f:
            pickle.dump(self.model, f)
        self.logger.info(f"Model saved to {self.model_save_path}")
