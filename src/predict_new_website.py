
import pandas as pd
import joblib
import os

# ==========================================
# STEP 1: FILE PATHS
# ==========================================

MODEL_PATH = "models/best_phishing_model.pkl"
TRAIN_PATH = "dataset/X_train.csv"
INPUT_PATH = "dataset/new_websites.csv"
OUTPUT_PATH = "reports/new_website_predictions.csv"

os.makedirs("reports", exist_ok=True)


# ==========================================
# STEP 2: CHECK REQUIRED FILES
# ==========================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Best model not found. Run hyperparameter_tuning.py first."
    )

if not os.path.exists(TRAIN_PATH):
    raise FileNotFoundError(
        "Training data not found. Run split_data.py first."
    )


# ==========================================
# STEP 3: LOAD MODEL AND FEATURE COLUMNS
# ==========================================

model = joblib.load(MODEL_PATH)

X_train = pd.read_csv(TRAIN_PATH)
feature_columns = X_train.columns.tolist()

print("Best phishing detection model loaded!")


# ==========================================
# STEP 4: CREATE NEW WEBSITE INPUT TEMPLATE
# ==========================================

if not os.path.exists(INPUT_PATH):

    # Create one sample row with the correct 30-feature format
    template = X_train.iloc[[0]].copy()

    template.to_csv(INPUT_PATH, index=False)

    print("\nNew website input template created!")
    print("File location:", INPUT_PATH)

    print("\nIMPORTANT:")
    print("1. Open dataset/new_websites.csv")
    print("2. Replace the sample values with new website feature values.")
    print("3. Keep all feature column names unchanged.")
    print("4. Save the file and run this script again.")

    raise SystemExit


# ==========================================
# STEP 5: LOAD NEW WEBSITE DATA
# ==========================================

new_data = pd.read_csv(INPUT_PATH)

if new_data.empty:
    raise ValueError("The new website input file is empty.")

print("\nNew website data loaded!")
print("Input shape:", new_data.shape)


# ==========================================
# STEP 6: VALIDATE FEATURES
# ==========================================

missing_columns = [
    col for col in feature_columns if col not in new_data.columns
]

extra_columns = [
    col for col in new_data.columns if col not in feature_columns
]

if missing_columns or extra_columns:

    print("\nFeature validation failed!")

    if missing_columns:
        print("Missing columns:", missing_columns)

    if extra_columns:
        print("Extra columns:", extra_columns)

    raise ValueError(
        "Input features must exactly match the training dataset."
    )

# Keep exactly the same feature order as training
new_data = new_data[feature_columns]

# Check numeric values
new_data = new_data.apply(pd.to_numeric, errors="coerce")

if new_data.isnull().any().any():
    raise ValueError(
        "Invalid or missing feature values found. "
        "Please enter valid numeric values."
    )


# ==========================================
# STEP 7: PREDICT NEW WEBSITES
# ==========================================

predictions = model.predict(new_data)

# Convert model output into readable labels
labels = []

for prediction in predictions:

    if prediction == -1:
        labels.append("Phishing Website")

    elif prediction == 1:
        labels.append("Legitimate Website")

    else:
        labels.append("Unknown")


# ==========================================
# STEP 8: CALCULATE PHISHING PROBABILITY
# ==========================================

phishing_probabilities = []

if hasattr(model, "predict_proba"):

    probabilities = model.predict_proba(new_data)

    # Find the probability column for class -1
    class_index = list(model.classes_).index(-1)

    phishing_probabilities = (
        probabilities[:, class_index] * 100
    ).round(2)

else:
    phishing_probabilities = [None] * len(predictions)


# ==========================================
# STEP 9: CREATE PREDICTION RESULTS
# ==========================================

results = pd.DataFrame({
    "Prediction": labels,
    "Phishing Probability (%)": phishing_probabilities
})

print("\n======================================")
print("NEW WEBSITE PREDICTION RESULTS")
print("======================================")

print(results.to_string(index=False))


# ==========================================
# STEP 10: SAVE RESULTS
# ==========================================

results.to_csv(OUTPUT_PATH, index=False)

print("\nPrediction completed successfully!")
print("Results saved at:", OUTPUT_PATH)
