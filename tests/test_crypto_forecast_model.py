
import yaml
import unittest
import os
import pandas as pd
from unittest.mock import patch, MagicMock

from src.crypto_forecast_model import CryptoForecastModel


class TestCryptoForecastModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load test config
        cls.test_config = {
            "general": {"random_state": 42},
            "model": {
                "library": "sklearn.ensemble",
                "name": "RandomForestClassifier",
                "model_params": {"n_estimators": 10},
            },
            "features": {"lags": "1,7", "windows": "3,5"},
            "training": {"test_size": 0.2, "cross_validation_folds": 5},
            "currency_mapping": {"BTC": 0, "ETH": 1},
            "chosen_trained_model": "test_model.pkl",
            "output": {"model_save_path": "test_model.pkl"},
        }

        # Create a temporary test configuration YAML
        current_file_path = os.path.abspath(__file__)
        src_dir = os.path.dirname(current_file_path)
        root_dir = os.path.dirname(src_dir)
        config_dir = os.path.join(root_dir, "config")
        cls.test_config_path = os.path.join(config_dir, "test_config.yaml")
        with open(cls.test_config_path, "w") as f:
            yaml.dump(cls.test_config, f)

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.test_config_path)

    def setUp(self):
        self.model = CryptoForecastModel(config_path=self.test_config_path)

        # Mock logger to prevent logging during tests
        self.model.logger = MagicMock()

    def test_initialization(self):
        self.assertEqual(self.model.random_state, 42)
        self.assertEqual(self.model.lags, [1, 7])
        self.assertEqual(self.model.windows, [3, 5])
        self.assertIn("BTC", self.model.crypto_mapping)

    @patch("CryptoForecastModel.extract_lags")
    @patch("CryptoForecastModel.extract_ma")
    def test_preprocess_and_combine_data(
        self,
        mock_extract_ma,
        mock_extract_lags
    ):
        # Mock preprocessing methods
        mock_extract_ma.side_effect = lambda df, windows: df
        mock_extract_lags.side_effect = lambda df, lags: df

        # Prepare sample data
        btc_data = pd.DataFrame({
            "price": [10, 11, 12, 13],
            "date": pd.date_range("2023-01-01", periods=4)
        }).set_index("date")

        eth_data = pd.DataFrame({
            "price": [20, 21, 22, 23],
            "date": pd.date_range("2023-01-01", periods=4)
        }).set_index("date")

        crypto_data_dict = {"BTC": btc_data, "ETH": eth_data}

        X, y = self.model.preprocess_and_combine_data(crypto_data_dict)
        self.assertIn("currency_id", X.columns)
        self.assertIn("price", y.index)

    def test_prepare_train_test_split(self):
        # Generate sample data
        X = pd.DataFrame({"feature1": range(10), "feature2": range(10)})
        y = pd.Series(range(10))

        X_train, X_test, y_train, y_test = self.model.prepare_train_test_split(X, y)
        self.assertEqual(len(X_train) + len(X_test), 10)
        self.assertEqual(len(y_train) + len(y_test), 10)

    @patch("sklearn.ensemble.RandomForestClassifier.fit")
    def test_train_model(self, mock_fit):
        # Mock fit method
        mock_fit.return_value = None

        X_train = pd.DataFrame({"feature1": range(10), "feature2": range(10)})
        y_train = pd.Series(range(10))

        self.model.train_model(X_train, y_train)
        mock_fit.assert_called_once()

    @patch("sklearn.ensemble.RandomForestClassifier.predict")
    def test_forecast(self, mock_predict):
        # Prepare sample data
        mock_predict.side_effect = [10, 11, 12]

        X = pd.DataFrame({
            "currency_id": [0] * 3,
            "price": [10, 11, 12],
            "lag_1": [9, 10, 11],
        }, index=pd.date_range("2023-01-01", periods=3))

        forecast_df = self.model.forecast(X, "BTC", future_days=3)
        self.assertEqual(len(forecast_df), 3)
        self.assertIn("predicted_price", forecast_df.columns)

    @patch("builtins.open", new_callable=unittest.mock.mock_open)
    @patch("pickle.load")
    def test_read_model(self, mock_pickle_load, mock_open):
        mock_pickle_load.return_value = MagicMock()
        self.model.read_model()
        mock_open.assert_called_once_with(self.model.model_load_path, "rb")

    @patch("builtins.open", new_callable=unittest.mock.mock_open)
    @patch("pickle.dump")
    def test_save_model(self, mock_pickle_dump, mock_open):
        self.model.save_model()
        mock_open.assert_called_once_with(self.model.model_save_path, "wb")
        mock_pickle_dump.assert_called_once_with(self.model.model, unittest.mock.ANY)


if __name__ == "__main__":
    unittest.main()

