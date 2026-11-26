
import pandas as pd
import joblib
import os

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score

# Create reports folder
os.makedirs("reports", exist_ok=True)

# Load only training data
X_train = pd.read_csv("dataset/X_train.csv")
y_train = pd.read_csv("dataset/y_train.csv").squeeze()

print("Training dataset loaded successfully!")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

# Define 5-Fold Cross Validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Define evaluation metrics
scoring = {
    "accuracy": "accuracy",
    "precision_phishing": make_scorer(
        precision_score, pos_label=-1, zero_division=0
    ),
    "recall_phishing": make_scorer(
        recall_score, pos_label=-1, zero_division=0
    ),
    "f1_phishing": make_scorer(
        f1_score, pos_label=-1, zero_division=0
    )
}

# Model file paths
model_files = {
    "Logistic Regression": "models/logistic_regression.pkl",
    "Decision Tree": "models/decision_tree.pkl",
    "Random Forest": "models/random_forest.pkl"
}

results = []

# Perform cross validation
for name, filepath in model_files.items():

    print("\n================================")
    print("Cross Validation:", name)
    print("================================")

    # Load model
    model = joblib.load(filepath)

    # Run 5-fold cross validation
    cv_results = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=1
    )

    # Calculate average scores
    accuracy = cv_results["test_accuracy"].mean()
    precision = cv_results["test_precision_phishing"].mean()
    recall = cv_results["test_recall_phishing"].mean()
    f1 = cv_results["test_f1_phishing"].mean()

    print("Fold Accuracy Scores:",
          cv_results["test_accuracy"])

    print("Average Accuracy:", round(accuracy, 4))
    print("Average Phishing Precision:", round(precision, 4))
    print("Average Phishing Recall:", round(recall, 4))
    print("Average Phishing F1-Score:", round(f1, 4))

    results.append({
        "Model": name,
        "CV Accuracy": accuracy,
        "CV Precision (Phishing)": precision,
        "CV Recall (Phishing)": recall,
        "CV F1-Score (Phishing)": f1
    })

# Create comparison table
results_df = pd.DataFrame(results)

print("\n========== CROSS VALIDATION SUMMARY ==========")
print(results_df.to_string(index=False))

# Save results
results_df.to_csv(
    "reports/cross_validation_results.csv",
    index=False
)

print("\n5-Fold Cross Validation completed successfully!")
print("Results saved in reports/cross_validation_results.csv")
