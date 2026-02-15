# preprocess.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import joblib

def preprocess_dataset(file_path="../data/telecom_customer_churn.csv"):
    df = pd.read_csv(file_path)

    columns_to_keep = ["City", "Contract", "Married", "Age", "Customer Status"]
    df = df[columns_to_keep].copy()

    df = df.drop_duplicates().reset_index(drop=True)

    df['City_original'] = df['City']
    df['Contract_original'] = df['Contract']

    df['Married'] = df['Married'].map({'Yes': 1, 'No': 0})

    contract_ohe = pd.get_dummies(df['Contract'], prefix='Contract')
    df = pd.concat([df.drop('Contract', axis=1), contract_ohe], axis=1)

    city_counts = df['City'].value_counts()

    frequent_cities = city_counts[city_counts > 16].index.tolist()

    df['City'] = df['City'].apply(lambda x: x if x in frequent_cities else "Other")

    city_ohe = pd.get_dummies(df['City'], prefix='City')
    df = pd.concat([df.drop('City', axis=1), city_ohe], axis=1)

    df['Customer Status'] = df['Customer Status'].map({'Churned': 1, 'Stayed': 0, 'Joined': 0})

    return df

if __name__ == "__main__":
    processed_df = preprocess_dataset()
    processed_df.to_csv("../preprocessed data/processed_churn_dataset.csv", index=False)
    print("✅ Preprocessing complete. Saved as 'processed_churn_dataset.csv'.")
