import os
import requests
import pandas as pd


class CryptoDataFetcher:
    def __init__(
            self,
            symbol,
            currency,
            limit
    ) -> None:
        self.symbol = symbol
        self.currency = currency
        self.limit = limit
        self.base_url = "https://min-api.cryptocompare.com/data/v2/histoday"
        self.api_key = os.getenv("CRYPTO_API_KEY")

    @property
    def api_key(self):
        """Getter for the API key"""
        return self._api_key

    @api_key.setter
    def api_key(self, value):
        """Setter for the API key"""
        if value is None or value == "":
            raise ValueError("API key cannot be None or empty.")
        self._api_key = value

    def fetch_data(self, save_data=False):
        """Fetch historical cryptocurrency data."""
        url = (
            f"{self.base_url}"
            f"?fsym={self.symbol}"
            f"&tsym={self.currency}"
            f"&limit={self.limit}"
        )
        response = requests.get(url)

        if response.status_code != 200:
            raise Exception(
                f"Error fetching data:\
                      {response.status_code} - {response.text}"
            )

        data = response.json()
        prices = data["Data"]["Data"]

        # Convert the response into a DataFrame
        df = pd.DataFrame(prices)
        df['date'] = pd.to_datetime(df['time'], unit='s')
        df = df[[
            'date', 'open', 'close', 'high', 'low', 'volumefrom', 'volumeto'
        ]]

        # Safe the dataframe if save_data is set to True
        if save_data:
            data_folder_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "data"
            )
            os.makedirs(data_folder_path, exist_ok=True)
            file_path = os.path.join(
                data_folder_path,
                f'{self.symbol}_in_{self.currency}_historical_data.csv'
            )
            df.to_csv(file_path, sep=",", index=False)
        return df
