import pandas as pd
import os

def preprocess_data():
    input_path = "../data/online_retail.xlsx"
    df = pd.read_excel(input_path)


    df = df.dropna(subset=["CustomerID"])

    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

    df = df.drop_duplicates()

    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    df["InvoiceYear"] = df["InvoiceDate"].dt.year
    df["InvoiceMonth"] = df["InvoiceDate"].dt.month
    df["InvoiceDay"] = df["InvoiceDate"].dt.day
    df["InvoiceHour"] = df["InvoiceDate"].dt.hour

    df["Country"] = df["Country"].astype("category").cat.codes

    output_folder = "./processed"
    output_path = "./processed/retail_processed.csv"

    os.makedirs(output_folder, exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"Processed data saved successfully to: {output_path}")
    print("Rows:", len(df))


if __name__ == "__main__":
    preprocess_data()
