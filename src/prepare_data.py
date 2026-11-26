
# ============================================
# MCA PBL PROJECT - SEC E
# PHISHING WEBSITE DETECTION
# STEP 6 - DATA PREPARATION
# ============================================

import os
import pandas as pd

# ============================================
# 1. CREATE DATASET FOLDER
# ============================================

os.makedirs("dataset", exist_ok=True)

# ============================================
# 2. LOAD ORIGINAL DATASET
# ============================================

df = pd.read_csv("dataset/phishing_websites.csv")

print("\n========== ORIGINAL DATASET ==========")
print("Original Shape:", df.shape)

# ============================================
# 3. CHECK TARGET COLUMN
# ============================================

target_column = "Result"

if target_column not in df.columns:
    raise ValueError(
        "Target column 'Result' not found. "
        f"Available columns: {df.columns.tolist()}"
    )

# ============================================
# 4. REMOVE IRRELEVANT INDEX COLUMN
# ============================================

df = df.loc[
    :,
    ~df.columns.str.contains("^Unnamed")
]

# ============================================
# 5. REMOVE DUPLICATE RECORDS
# ============================================

print("\n========== DUPLICATE REMOVAL ==========")

duplicates = df.duplicated().sum()

print("Duplicates found:", duplicates)

df = df.drop_duplicates().copy()

print("Shape after removing duplicates:", df.shape)

# ============================================
# 6. HANDLE MISSING VALUES
# ============================================

print("\n========== MISSING VALUES ==========")

print(df.isnull().sum())

# Separate target and features
X = df.drop(columns=[target_column]).copy()
y = df[target_column].copy()

# Remove rows where target is missing
valid_rows = y.notna()

X = X.loc[valid_rows].copy()
y = y.loc[valid_rows].copy()

# Fill missing numerical features with median
numeric_columns = X.select_dtypes(
    include=["number"]
).columns

for column in numeric_columns:
    X[column] = X[column].fillna(
        X[column].median()
    )

# Fill missing categorical features with mode
categorical_columns = X.select_dtypes(
    exclude=["number"]
).columns

for column in categorical_columns:
    if X[column].isnull().any():
        mode_value = X[column].mode()

        if not mode_value.empty:
            X[column] = X[column].fillna(
                mode_value[0]
            )
        else:
            X[column] = X[column].fillna("Unknown")

print("\nMissing values handled successfully!")

# ============================================
# 7. ENCODE CATEGORICAL FEATURES
# ============================================

print("\n========== CATEGORICAL ENCODING ==========")

print("Categorical columns:", list(categorical_columns))

if len(categorical_columns) > 0:

    X = pd.get_dummies(
        X,
        columns=list(categorical_columns),
        drop_first=False,
        dtype=int
    )

print("Encoding completed!")

# ============================================
# 8. CHECK TARGET VALUES
# ============================================

print("\n========== TARGET DISTRIBUTION ==========")

print(y.value_counts())

print("\nTarget classes:", y.unique())

# ============================================
# 9. FINAL DATASET VALIDATION
# ============================================

print("\n========== FINAL VALIDATION ==========")

print("Features Shape:", X.shape)
print("Target Shape:", y.shape)

print("Remaining Missing Values:", X.isnull().sum().sum())

print("Duplicate Records:", X.duplicated().sum())

# Ensure all features are numeric
if not all(pd.api.types.is_numeric_dtype(dtype)
           for dtype in X.dtypes):
    raise ValueError("Some features are not numeric!")

# Check target
if y.nunique() != 2:
    raise ValueError("Expected a binary classification target!")

# ============================================
# 10. SAVE PREPARED DATASET
# ============================================

X.to_csv(
    "dataset/X_features.csv",
    index=False
)

y.to_csv(
    "dataset/y_target.csv",
    index=False,
    header=True
)

prepared_df = X.copy()
prepared_df[target_column] = y.values

prepared_df.to_csv(
    "dataset/prepared_dataset.csv",
    index=False
)

# ============================================
# 11. FINAL OUTPUT
# ============================================

print("\n===================================")
print("DATA PREPARATION COMPLETED!")
print("===================================")

print("Original dataset:", df.shape)
print("Prepared features:", X.shape)
print("Prepared target:", y.shape)

print("\nSaved Files:")
print("1. dataset/X_features.csv")
print("2. dataset/y_target.csv")
print("3. dataset/prepared_dataset.csv")

print("\nDataset is ready for Machine Learning!")
