from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from src.data_preprocessing import preprocess

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    return rmse, r2


def main():
    X_train, X_test, y_train, y_test, scaler = preprocess()

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        rmse, r2 = evaluate(model, X_test, y_test)
        results[name] = (rmse, r2)
        print(f"{name:20s} RMSE: {rmse:.4f}  R²: {r2:.4f}")

    # Lower RMSE is better
    best_name = min(results, key=lambda n: results[n][0])
    print(f"\nBest model: {best_name}")

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(models[best_name], MODELS_DIR / "model.pkl", compress=3)
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    print(f"Saved model.pkl and scaler.pkl to {MODELS_DIR}")


if __name__ == "__main__":
    main()