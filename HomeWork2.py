import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score


def load_data():
    """Load the California housing dataset."""
    housing = fetch_california_housing()
    X = housing.data  # features
    y = housing.target  # target variable
    feature_names = housing.feature_names
    return X, y, feature_names


def train_model(X_train, y_train):
    """Train a linear regression model."""
    # Initialize and fit the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model on test data."""
    # Predict and calculate Mean Squared Error
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)
    return mse


def perform_cross_validation(X_train, y_train):
    """Perform cross-validation to evaluate the model."""
    # Initialize and perform cross-validation
    model = LinearRegression()
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    cv_scores = -cv_scores  # Convert negative MSE to positive
    print("Cross-Validation Scores:", cv_scores)  # [score1, score2, ..., score5]
    print("Mean Cross-Validation Score:", np.mean(cv_scores))  # mean_score
    return cv_scores


def main():
    # Load the data
    X, y, feature_names = load_data()
    df = pd.DataFrame(X, columns=feature_names)
    df["Target"] = y

    # Simple EDA
    print("Basic Statistics:")
    print(df.describe())

    # Correlation Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix")
    plt.show()

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    mse = evaluate_model(model, X_test, y_test)

    # Perform cross-validation
    cv_scores = perform_cross_validation(X_train, y_train)


if __name__ == "__main__":
    main()
