from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
TARGET = "MedHouseVal"


def load_data() -> pd.DataFrame:
    """Load the California Housing dataset and save a local CSV backup."""
    df = fetch_california_housing(as_frame=True).frame
    (ROOT / "data").mkdir(exist_ok=True)
    df.to_csv(ROOT / "data" / "raw_data.csv", index=False)
    return df


def preprocess(test_size: float = 0.2, random_state: int = 42):
    """Split into train/test sets and scale the features.

    Returns: X_train_scaled, X_test_scaled, y_train, y_test, scaler
    """
    df = load_data().dropna()

    X = df.drop(columns=TARGET)
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Fit on training data only, so no information from the test set leaks in
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler = preprocess()
    print("Train:", X_train.shape, "Test:", X_test.shape)
    print("Scaled feature means (should be ~0):", X_train.mean(axis=0).round(2))