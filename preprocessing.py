import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
def load_data(path):
    """
    Load the customer churn dataset.
    """
    df = pd.read_csv(path)
    return df
def preprocess_data(df):
    """
    Clean and preprocess the dataset.
    """
    
    if "CustomerID" in df.columns:
        df.drop("CustomerID", axis=1, inplace=True)
    encoder = LabelEncoder()

    categorical_columns = [
        "Gender",
        "Subscription Type",
        "Contract Length"
    ]

    for column in categorical_columns:
        if column in df.columns:
            df[column] = encoder.fit_transform(df[column])

    return df


def split_dataset(df):

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    return X_train, X_test, y_train, y_test


def scale_data(X_train, X_test):

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, scaler


if __name__ == "__main__":

    print("Loading dataset...")

    df = load_data("dataset/churn.csv")

    print("\nFirst 5 rows:\n")
    print(df.head())

    print("\nDataset Shape:")
    print(df.shape)

    df = preprocess_data(df)

    X_train, X_test, y_train, y_test = split_dataset(df)

    X_train, X_test, scaler = scale_data(
        X_train,
        X_test
    )

    print("\nPreprocessing Completed Successfully!")

    print("\nTraining Samples :", len(X_train))
    print("Testing Samples  :", len(X_test))