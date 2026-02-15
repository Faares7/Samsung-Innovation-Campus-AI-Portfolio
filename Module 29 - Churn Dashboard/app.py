# app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np

@st.cache_data
def load_data():
    df = pd.read_csv("preprocessed data/processed_churn_dataset.csv")
    return df

@st.cache_resource
def load_model_objects():
    model = joblib.load("models/churn_model.pkl")
    scaler = joblib.load("models/age_scaler.pkl")
    feature_cols = joblib.load("models/feature_columns.pkl")
    return model, scaler, feature_cols

df = load_data()
model, scaler, feature_cols = load_model_objects()

st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", ["Dashboard", "Predict Churn"])

if page == "Dashboard":
    st.title("Telecom Churn Dashboard")

    st.sidebar.subheader("Filters")
    selected_cities = st.sidebar.multiselect(
        "Select City",
        options=df["City_original"].unique(),
        default=df["City_original"].unique()
    )
    selected_contract = st.sidebar.radio(
        "Select Contract",
        options=df["Contract_original"].unique()
    )
    selected_married = st.sidebar.radio(
        "Select Married Status",
        options=[0,1],
        format_func=lambda x: "Yes" if x==1 else "No"
    )
    min_age, max_age = int(df["Age"].min()), int(df["Age"].max())
    age_range = st.sidebar.slider(
        "Select Age Range",
        min_value=min_age,
        max_value=max_age,
        value=(min_age, max_age)
    )

    filtered_df = df[
        (df["City_original"].isin(selected_cities)) &
        (df["Contract_original"] == selected_contract) &
        (df["Married"] == selected_married) &
        (df["Age"] >= age_range[0]) &
        (df["Age"] <= age_range[1])
    ]

    st.subheader(f"Filtered Data: {filtered_df.shape[0]} customers")
    st.dataframe(filtered_df.head(10))

    st.subheader("Churn by City")
    churn_by_city = filtered_df.groupby("City_original")["Customer Status"].mean().sort_values(ascending=False)
    st.bar_chart(churn_by_city)

    st.subheader("Churn by Contract")
    churn_by_contract = filtered_df.groupby("Contract_original")["Customer Status"].mean()
    st.bar_chart(churn_by_contract)

    st.subheader("Churn by Age")
    st.bar_chart(filtered_df.groupby("Age")["Customer Status"].mean())

else:
    st.title("Predict Churn Probability")

    user_city = st.selectbox("City", options=df["City_original"].unique())
    user_contract = st.selectbox("Contract", options=df["Contract_original"].unique())
    user_married = st.selectbox("Married Status", options=[0,1], format_func=lambda x: "Yes" if x==1 else "No")
    user_age = st.number_input("Age", min_value=int(df["Age"].min()), max_value=int(df["Age"].max()), value=30)

    if st.button("Predict"):
        input_dict = {"Age": [user_age], "Married": [user_married]}

        for contract in [c for c in feature_cols if c.startswith("Contract_")]:
            input_dict[contract] = [1 if contract == f"Contract_{user_contract}" else 0]

        for city in [c for c in feature_cols if c.startswith("City_")]:
            city_name = city.replace("City_", "")
            city_value = user_city if user_city in city_name else "Other"
            input_dict[city] = [1 if city_name == city_value else 0]

        input_df = pd.DataFrame(input_dict)

        input_df[["Age"]] = scaler.transform(input_df[["Age"]])

        input_df = input_df.reindex(columns=feature_cols, fill_value=0)

        churn_prob = model.predict_proba(input_df)[:, 1][0]

        st.success(f"Churn Probability: {churn_prob*100:.2f}%")
