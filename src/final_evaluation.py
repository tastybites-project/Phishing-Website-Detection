
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

# ==========================================
# STEP 1: CREATE REPORTS FOLDER
# ==========================================

os.makedirs("reports", exist_ok=True)


# ==========================================
# STEP 2: LOAD TEST DATA
# ==========================================

X_test = pd.read_csv("dataset/X_test.csv")
y_test = pd.read_csv("dataset/y_test.csv").squeeze()

print("Testing dataset loaded successfully!")
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# ==========================================
# STEP 3: LOAD BEST TUNED MODEL
# ==========================================

model_path = "models/best_phishing_model.pkl"

if not os.path.exists(model_path):
    raise FileNotFoundError(
        "Best model not found! Run hyperparameter_tuning.py first."
    )

best_model = joblib.load(model_path)

print("\nBest tuned model loaded successfully!")


# ==========================================
# STEP 4: PREDICT TEST DATA
# ==========================================

y_pred = best_model.predict(X_test)

print("\nPrediction completed successfully!")


# ==========================================
# STEP 5: CALCULATE EVALUATION METRICS
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    pos_label=-1,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    pos_label=-1,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    pos_label=-1,
    zero_division=0
)


# ==========================================
# STEP 6: DISPLAY FINAL RESULTS
# ==========================================

print("\n======================================")
print("FINAL MODEL EVALUATION RESULTS")
print("======================================")

print("Accuracy:", round(accuracy, 4))
print("Phishing Precision:", round(precision, 4))
print("Phishing Recall:", round(recall, 4))
print("Phishing F1-Score:", round(f1, 4))


# ==========================================
# STEP 7: CLASSIFICATION REPORT
# ==========================================

report = classification_report(
    y_test,
    y_pred,
    labels=[-1, 1],
    target_names=["Phishing", "Legitimate"],
    zero_division=0
)

print("\nClassification Report:")
print(report)

with open(
    "reports/final_classification_report.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(report)


# ==========================================
# STEP 8: CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[-1, 1]
)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Phishing", "Legitimate"],
    yticklabels=["Phishing", "Legitimate"]
)

plt.title("Best Model - Final Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.tight_layout()

plt.savefig(
    "reports/final_confusion_matrix.png",
    dpi=300
)

plt.close()


# ==========================================
# STEP 9: SAVE FINAL METRICS
# ==========================================

final_results = pd.DataFrame([{
    "Accuracy": accuracy,
    "Phishing Precision": precision,
    "Phishing Recall": recall,
    "Phishing F1-Score": f1
}])

final_results.to_csv(
    "reports/final_model_results.csv",
    index=False
)


# ==========================================
# STEP 10: COMPLETION
# ==========================================

print("\n======================================")
print("FINAL EVALUATION COMPLETED!")
print("======================================")

print("Final results saved successfully!")

print("1. reports/final_model_results.csv")
print("2. reports/final_classification_report.txt")
print("3. reports/final_confusion_matrix.png")
