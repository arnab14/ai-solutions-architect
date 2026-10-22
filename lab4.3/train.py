import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

# Load and preprocess data
def load_and_preprocess_data(data_path):
    data = pd.read_csv(data_path)
    
    # Drop unnecessary columns
    data = data.drop(['customerID'], axis=1)
    
    # Convert TotalCharges to numeric, setting errors='coerce' to turn invalid parsing into NaN
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    
    # Fill NaN values in TotalCharges with the mean (or another strategy)
    data['TotalCharges'].fillna(data['TotalCharges'].mean(), inplace=True)

    # Encode the target variable to numerical format
    label_encoder = LabelEncoder()
    data['Churn'] = label_encoder.fit_transform(data['Churn'])

    # One-hot encode categorical features
    categorical_cols = data.select_dtypes(include=['object']).columns
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
    
    return data

# Split data into features (X) and target (y)
def split_data(data, target_column):
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    return X, y

if __name__ == "__main__":
    data_path = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

    # Check if the data file exists
    if not os.path.isfile(data_path):
        print(f"Error: Data file not found at {data_path}")
        exit()

    # Load and preprocess the data
    try:
        data = load_and_preprocess_data(data_path)
    except Exception as e:
        print(f"Error preprocessing data: {e}")
        exit()

    # Split the data
    try:
        X, y = split_data(data, "Churn")
    except Exception as e:
        print(f"Error splitting data: {e}")
        exit()

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale numerical features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Start an MLflow run
    with mlflow.start_run():
        # Define hyperparameters
        params = {
            "penalty": "l2",
            "C": 0.1,
            "solver": "liblinear",
            "random_state": 42
        }

        # Log parameters
        mlflow.log_params(params)

        # Train the model
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)

        # Log the model
        mlflow.sklearn.log_model(model, "model")

        # Print out the model metrics
        print(f"Model accuracy: {accuracy}")
        print(f"Model precision: {precision}")
        print(f"Model recall: {recall}")
        print(f"Model f1-score: {f1}")