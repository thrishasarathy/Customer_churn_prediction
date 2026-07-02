import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from preprocessing import (
    load_data,
    preprocess_data,
    split_dataset,
    scale_data
)
print("Loading dataset...")

df = load_data("dataset/churn.csv")
print("Cleaning dataset...")

df = preprocess_data(df)
X_train, X_test, y_train, y_test = split_dataset(df)

X_train, X_test, scaler = scale_data(
    X_train,
    X_test
)



print("\nTraining Logistic Regression...")

lr = LogisticRegression(max_iter=1000)

lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

lr_acc = accuracy_score(y_test, lr_pred)

print("\nAccuracy :", lr_acc)

print("\nClassification Report")

print(classification_report(y_test, lr_pred))

print("\nConfusion Matrix")

print(confusion_matrix(y_test, lr_pred))



print("\nTraining Random Forest...")

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

rf_acc = accuracy_score(y_test, rf_pred)

print("\nAccuracy :", rf_acc)

print("\nClassification Report")

print(classification_report(y_test, rf_pred))

print("\nConfusion Matrix")

print(confusion_matrix(y_test, rf_pred))


if rf_acc > lr_acc:

    print("\nRandom Forest is Better.")

    best_model = rf

else:

    print("\nLogistic Regression is Better.")

    best_model = lr



os.makedirs("models", exist_ok=True)

joblib.dump(best_model, "models/churn_model.pkl")

print("\nModel Saved Successfully!")

print("\nLocation : models/churn_model.pkl")