# train.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os

def train_churn_model(preprocessed_file="../preprocessed data/processed_churn_dataset.csv"):

    df = pd.read_csv(preprocessed_file)
    print(f"✅ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")


    target_col = "Customer Status"

    X = df.drop([target_col, "City_original", "Contract_original"], axis=1)
    y = df[target_col]


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"✅ Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    age_col = "Age"
    scaler = StandardScaler()
    X_train[age_col] = scaler.fit_transform(X_train[[age_col]])
    X_test[age_col] = scaler.transform(X_test[[age_col]])
    print("✅ Age column scaled")

    joblib.dump(scaler, "../models/age_scaler.pkl")
    print("✅ Scaler saved: models/age_scaler.pkl")

    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)
    print("✅ Logistic Regression model trained")

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("ROC-AUC Score:", roc_auc_score(y_test, y_pred_proba))

    joblib.dump(model, "../models/churn_model.pkl")
    joblib.dump(X_train.columns.tolist(), "../models/feature_columns.pkl")
    print("✅ Model saved: models/churn_model.pkl")
    print("✅ Feature columns saved: models/feature_columns.pkl")

    return model, scaler, X_train.columns.tolist()


if __name__ == "__main__":
    train_churn_model()
