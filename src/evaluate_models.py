
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Create reports folder
os.makedirs("reports", exist_ok=True)

# Load testing dataset
X_test = pd.read_csv("dataset/X_test.csv")
y_test = pd.read_csv("dataset/y_test.csv").squeeze()

# Model file paths
model_files = {
    "Logistic Regression": "models/logistic_regression.pkl",
    "Decision Tree": "models/decision_tree.pkl",
    "Random Forest": "models/random_forest.pkl"
}

results = []

# Evaluate each model
for name, filepath in model_files.items():

    print("\n================================")
    print("Evaluating:", name)
    print("================================")

    # Load trained model
    model = joblib.load(filepath)

    # Predict testing data
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=-1, zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label=-1, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=-1, zero_division=0)

    print("Accuracy:", round(accuracy, 4))
    print("Precision (Phishing):", round(precision, 4))
    print("Recall (Phishing):", round(recall, 4))
    print("F1-Score (Phishing):", round(f1, 4))

    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        labels=[-1, 1],
        target_names=["Phishing", "Legitimate"],
        zero_division=0
    ))

    # Save metrics
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    })

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=[-1, 1])

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Phishing", "Legitimate"],
        yticklabels=["Phishing", "Legitimate"]
    )

    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()

    filename = name.lower().replace(" ", "_")
    plt.savefig(f"reports/{filename}_confusion_matrix.png")
    plt.close()

# Create comparison table
results_df = pd.DataFrame(results)

print("\n========== MODEL COMPARISON ==========")
print(results_df.to_string(index=False))

# Save results
results_df.to_csv("reports/model_evaluation_results.csv", index=False)

print("\nEvaluation completed successfully!")
print("Results saved in reports folder.")

