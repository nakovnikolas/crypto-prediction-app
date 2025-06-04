# 📈 Crypto Price Prediction App

This repository contains a work-in-progress Streamlit app for predicting cryptocurrency prices using machine learning. The project fetches historical price data via a public crypto API, trains multiple regression models, and displays predictions and evaluation metrics in an interactive web dashboard.

📅 **Status**: Under development  
🧠 **Best Model So Far**: Random Forest Regressor  

---

## 🔍 Project Overview

This project aims to explore whether recent trends in cryptocurrency prices can be used to predict short-term future prices. The application is designed as a full ML workflow:

- Fetches real-time or historical data from a cryptocurrency API
- Preprocesses data and engineers relevant time-series features
- Trains and evaluates multiple ML regression models
- Deploys the best model (currently Random Forest) through a Streamlit app

---

## 🧪 Features

- 📡 **API Integration**: Automatically pulls cryptocurrency prices from a public API
- 🧠 **Machine Learning**: Random Forest regressor, XGBoost regressor and Prophet model tested
- 📊 **Evaluation Metrics**: MSE, RMSE, and visualizations of predictions vs. actual values
- 🌐 **Streamlit UI**: Interactive dashboard for viewing model predictions and summaries

---

## 🗂️ Repository Structure
```text
crypto-prediction-app/
|-- config/
|   |-- config.yaml
|
|-- data/
|   |-- BTC_in_USD_historical_data.csv
|   |-- ETH_in_USD_historical_data.csv
|   |-- LTC_in_USD_historical_data.csv
|
|-- models/
|   |-- rf_01.pkl
|   |-- rf_02.pkl
|   |-- rf_03.pkl
|
|-- notebooks/
|   |-- training_experiments.ipynb
|
|-- src/
|   |-- __init__.py
|   |-- app.py
|   |-- config_loader.py
|   |-- crypto_forecast_model.py
|   |-- fetch_crypto_data.py
|   |-- lag_utility_functions.py
|   |-- logger_manager.py
|   |-- ma_utility_functions.py
|   |-- model_utility_functions.py
|
|-- tests/
|   |-- __init__.py
|   |-- test_config_loader.py
|   |-- test_crypto_data_fetcher.py
|   |-- test_crypto_forecast_model.py
|
|-- run.py
|-- requirements.txt
|-- .gitignore
|-- LICENSE
|-- README.md
```


⚙️ Setup Instructions
To run the project locally, follow these steps:

1. Clone the repository
```
git clone https://github.com/nakovnikolas/crypto-prediction-app.git
cd crypto-prediction-app
```

2. Create and activate a virtual environment
macOS/Linux:
```
python3 -m venv venv
source venv/bin/activate
```

Windows (CMD):
```
python -m venv venv
venv\Scripts\activate
```

3. Install the dependencies
```
pip install -r requirements.txt
```

4. Run the Streamlit app
```
streamlit run app/app.py
```
Once running, open your browser to http://localhost:8501 to use the app.

📦 Technologies Used

Language:\
Python 3.12.7

Libraries:\
pandas, numpy, scikit-learn, matplotlib, seaborn, xgboost, joblib,streamlit,
requests or similar (for API calls)

🚧 Roadmap

Planned improvements:\
[ ] Finish developing the Streamlit app.
[ ] Implement an LLM that adds an additional feature using X tweets on crypto.

📬 Contact\
Nikolas Nakov
📧 Email: nikolas.nakov@gmail.com\
🔗 LinkedIn: https://www.linkedin.com/in/nikolas-nakov-60880515b/