# src/train.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

DATA_PATH = "processed/retail_processed.csv"

def main():

    # 1. Load processed data
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Processed data not found at: {DATA_PATH}\n"
                                "Run preprocess.py first to create processed/retail_processed.csv")

    df = pd.read_csv(DATA_PATH)

    # 2. Define target + features
    # Here we predict the TotalPrice (per-row). Adjust if you want invoice-level aggregation.
    if "TotalPrice" not in df.columns:
        raise KeyError("Expected column 'TotalPrice' not found in processed data.")

    y = df["TotalPrice"]
    X = df[["Quantity", "UnitPrice", "InvoiceMonth", "Country"]]

    # 3. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4. ML pipeline
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(n_estimators=150, random_state=42))
    ])

    # 5. Train
    print("Training model...")
    model.fit(X_train, y_train)

    # 6. Evaluate
    preds = model.predict(X_test)

    # safety: ensure shapes align
    if len(preds) != len(y_test):
        raise ValueError(f"Prediction length ({len(preds)}) != y_test length ({len(y_test)})")

    mse = mean_squared_error(y_test, preds)  # returns MSE
    rmse = mse ** 0.5
    print(f"RMSE: {rmse:.4f}")
    print("R²:", r2_score(y_test, preds))

    # 7. Save model
    save_path = "model.joblib"
    joblib.dump(model, save_path)

    print(f"\nModel saved successfully at: {save_path}")


if __name__ == "__main__":
    main()
