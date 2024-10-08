import os
import unittest
from unittest.mock import patch, MagicMock

import pandas as pd

from src.fetch_crypto_data import CryptoDataFetcher


class TestCryptoDataFetcher(unittest.TestCase):

    @patch.dict(os.environ, {"CRYPTO_API_KEY": "test_api_key"})
    def setUp(self):
        self.symbol = "BTC"
        self.currency = "USD"
        self.limit = 10
        self.fetcher = CryptoDataFetcher(
            self.symbol,
            self.currency,
            self.limit
        )

    def test_initialization(self):
        """Test that the CryptoDataFetcher initializes correctly."""
        self.assertEqual(self.fetcher.symbol, self.symbol)
        self.assertEqual(self.fetcher.currency, self.currency)
        self.assertEqual(self.fetcher.limit, self.limit)
        self.assertEqual(self.fetcher.api_key, "test_api_key")

    def test_api_key_validation(self):
        """Test that a ValueError is raised for an empty API key."""
        with self.assertRaises(ValueError):
            self.fetcher.api_key = ""

    @patch('src.crypto_data_fetcher.requests.get')
    def test_fetch_data(self, mock_get):
        """Test the fetch_data method with a mocked API response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "Data": {
                "Data": [
                    {
                        "time": 1638316800,
                        "open": 1,
                        "close": 2,
                        "high": 3,
                        "low": 0.5,
                        "volumefrom": 100,
                        "volumeto": 200
                    }
                ]
            }
        }
        mock_get.return_value = mock_response

        df = self.fetcher.fetch_data()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape)
