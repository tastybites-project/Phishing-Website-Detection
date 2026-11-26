
import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Create dataset folder if it does not exist
os.makedirs("dataset", exist_ok=True)

# Load prepared features and target
X = pd.read_csv("dataset/X_features.csv")
y = pd.read_csv("dataset/y_target.csv").squeeze()

print("Original Dataset Shape:", X.shape)
print("Target Shape:", y.shape)

# Check row counts
if len(X) != len(y):
    raise ValueError("Features and target row counts do not match!")

# Split dataset into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Save split datasets
X_train.to_csv("dataset/X_train.csv", index=False)
X_test.to_csv("dataset/X_test.csv", index=False)

y_train.to_csv("dataset/y_train.csv", index=False)
y_test.to_csv("dataset/y_test.csv", index=False)

# Display results
print("\nDataset Split Completed Successfully!")

print("\nTraining Features Shape:", X_train.shape)
print("Testing Features Shape:", X_test.shape)

print("\nTraining Target Distribution:")
print(y_train.value_counts())

print("\nTesting Target Distribution:")
print(y_test.value_counts())

print("\nAll split files saved inside dataset folder.")
