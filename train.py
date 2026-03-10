# train.py

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib


def train_model():

    # Load dataset
    data = load_iris()
    X = data.data
    y = data.target

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Calculate accuracy
    accuracy = model.score(X_test, y_test)
    print("Model Accuracy:", accuracy)

    # Save model
    joblib.dump(model, "model.pkl")
    print("Model saved as model.pkl")


if __name__ == "__main__":
    print("Training started...")
    train_model()