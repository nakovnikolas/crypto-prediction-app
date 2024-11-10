import pandas as pd
import yaml
import importlib

from model_utility_functions import preprocess_data
from lag_utility_functions import extract_lags
from ma_utility_functions import extract_ma


class CryptoForecastModel:
    def __init__(self, config_path="config.yaml"):
        # Load configuration from YAML
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

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