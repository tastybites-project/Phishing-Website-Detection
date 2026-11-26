
import pandas as pd
import os
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Create models folder
os.makedirs("models", exist_ok=True)

# Load training dataset
X_train = pd.read_csv("dataset/X_train.csv")
y_train = pd.read_csv("dataset/y_train.csv").squeeze()

print("Training data loaded successfully!")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

# Define three machine learning models

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000, random_state=42))
    ]),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
}

# Train and save each model
for name, model in models.items():

    print("\nTraining:", name)

    model.fit(X_train, y_train)

    # Training accuracy (initial check only)
    y_pred = model.predict(X_train)
    accuracy = accuracy_score(y_train, y_pred)

    # Save trained model
    filename = name.lower().replace(" ", "_") + ".pkl"
    filepath = os.path.join("models", filename)

    joblib.dump(model, filepath)

    print("Training Accuracy:", round(accuracy * 100, 2), "%")
    print("Model saved:", filepath)

print("\nAll three models trained successfully!")
