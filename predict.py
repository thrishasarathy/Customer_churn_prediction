import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
model = joblib.load("models/churn_model.pkl")
gender_encoder = LabelEncoder()
gender_encoder.fit(["Female", "Male"])
subscription_encoder = LabelEncoder()
subscription_encoder.fit(["Basic", "Standard", "Premium"])

contract_encoder = LabelEncoder()
contract_encoder.fit(["Monthly", "Quarterly", "Annual"])
print("\n========== Customer Churn Prediction ==========\n")
age = int(input("Enter Age: "))
gender = input("Enter Gender (Male/Female): ").strip().title()
tenure = int(input("Enter Tenure: "))
usage = int(input("Enter Usage Frequency: "))
support = int(input("Enter Support Calls: "))
delay = int(input("Enter Payment Delay: "))
subscription = input("Enter Subscription Type (Basic/Standard/Premium): ").strip().title()
contract = input("Enter Contract Length (Monthly/Quarterly/Annual): ").strip().title()
spend = float(input("Enter Total Spend: "))
interaction = int(input("Enter Last Interaction: "))
gender = gender_encoder.transform([gender])[0]
subscription = subscription_encoder.transform([subscription])[0]
contract = contract_encoder.transform([contract])[0]
customer = pd.DataFrame([{
    "Age": age,
    "Gender": gender,
    "Tenure": tenure,
    "Usage Frequency": usage,
    "Support Calls": support,
    "Payment Delay": delay,
    "Subscription Type": subscription,
    "Contract Length": contract,
    "Total Spend": spend,
    "Last Interaction": interaction
}])
prediction = model.predict(customer)
probability = model.predict_proba(customer)
print("\n==============================")

if prediction[0] == 1:
    print("Prediction : Customer WILL Churn")
else:
    print("Prediction : Customer will NOT Churn")

print("\nProbability")

print("Stay :", round(probability[0][0] * 100, 2), "%")

print("Churn:", round(probability[0][1] * 100, 2), "%")

print("\n==============================")