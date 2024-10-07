from fetch_crypto_data import CryptoDataFetcher


fetcher = CryptoDataFetcher("BTC", "USD", 365)
fetcher.fetch_data(save_data=True)
