
import yaml
import unittest
from unittest.mock import patch, mock_open

from test_crypto_forecast_model import CryptoForecastModel


class TestCryptoForecastModel(unittest.TestCase):

    def setUp(self):
        """Set up the necessary state before each test method."""
        # Mock the config YAML for testing
        self.mock_config = {
            "general": {"random_state": 42},
            "model": {
                "library": "sklearn",
                "name": "RandomForestClassifier",
                "model_params": {
                    "n_estimators": 100,
                    "max_depth": 10
                }
            },
            "features": {
                "lags": [1, 7, 10],
                "windows": [3, 5, 7]
            },
            "training": {
                "test_size": 0.2,
                "cross_validation_folds": 5
            },
            "output": {"model_save_path": "models/best_model.pkl"}
        }

        # Mock the loading of the YAML file
        with patch(
            "builtins.open",
            mock_open(read_data=yaml.dump(self.mock_config))
        ):
            self.model = CryptoForecastModel(config_path="config.yaml")

    def test_get_model_class
