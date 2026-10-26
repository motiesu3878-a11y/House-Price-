# House Price Predictor

A web app that predicts house prices in California using machine learning. Built with Python, Scikit-Learn and Streamlit.

## What it does

- Trains two models (Linear Regression and Random Forest) and keeps the best one
- Lets you enter house details with sliders and get a price estimate

## Results

| Model | RMSE | R² |
|---|---|---|
| Linear Regression | 0.7456 | 0.5758 |
| Random Forest | 0.5059 | 0.8047 |

## How to run

1. Install the packages: `pip install -r requirements.txt`
2. Train the model: `python -m src.train`
3. Start the app: `python -m streamlit run app.py`

## Dataset

California Housing dataset from scikit-learn (1990 census data).

