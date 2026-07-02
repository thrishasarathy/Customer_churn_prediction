import streamlit as st
import joblib
import pandas as pd
model = joblib.load("models/churn_model.pkl")

st.set_page_config(page_title="Customer Churn Prediction")

st.title("Customer Churn Prediction System")

st.write("Enter Customer Details")

age = st.number_input("Age", 18, 100, 30)

gender = st.selectbox("Gender", ["Female", "Male"])

tenure = st.number_input("Tenure", 0, 100, 12)

usage = st.number_input("Usage Frequency", 0, 100, 15)

support = st.number_input("Support Calls", 0, 20, 2)

delay = st.number_input("Payment Delay", 0, 50, 5)

subscription = st.selectbox(
    "Subscription Type",
    ["Basic", "Standard", "Premium"]
)

contract = st.selectbox(
    "Contract Length",
    ["Monthly", "Quarterly", "Annual"]
)

spend = st.number_input("Total Spend", 0.0, 100000.0, 1500.0)

interaction = st.number_input("Last Interaction", 0, 100, 10)
gender = 1 if gender == "Male" else 0

subscription_dict = {
    "Basic": 0,
    "Standard": 1,
    "Premium": 2
}

contract_dict = {
    "Monthly": 0,
    "Quarterly": 1,
    "Annual": 2
}
subscription = subscription_dict[subscription]
contract = contract_dict[contract]

if st.button("Predict"):

    customer = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "Tenure": [tenure],
        "Usage Frequency": [usage],
        "Support Calls": [support],
        "Payment Delay": [delay],
        "Subscription Type": [subscription],
        "Contract Length": [contract],
        "Total Spend": [spend],
        "Last Interaction": [interaction]
    })

    prediction = model.predict(customer)

    probability = model.predict_proba(customer)

    if prediction[0] == 1:
        st.error("Customer is likely to Churn")
    else:
        st.success("Customer is likely to Stay")

    st.write("### Probability")

    st.write(
        f"Stay : {round(probability[0][0]*100,2)} %"
    )

    st.write(
        f"Churn : {round(probability[0][1]*100,2)} %"
    )