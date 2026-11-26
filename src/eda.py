
# ============================================
# MCA PBL PROJECT - SEC E
# PHISHING WEBSITE DETECTION
# STEP 5 - EXPLORATORY DATA ANALYSIS (EDA)
# ============================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================
# 1. CREATE REPORTS FOLDER
# ============================================

os.makedirs("reports", exist_ok=True)

# ============================================
# 2. LOAD DATASET
# ============================================

df = pd.read_csv("dataset/phishing_websites.csv")

print("\n===================================")
print("PHISHING WEBSITE DETECTION - EDA")
print("===================================")

print("\nDataset loaded successfully!")

# ============================================
# 3. FIRST FIVE ROWS
# ============================================

print("\n========== FIRST FIVE ROWS ==========")
print(df.head())

# ============================================
# 4. DATASET SHAPE
# ============================================

print("\n========== DATASET SHAPE ==========")
print("Total Rows:", df.shape[0])
print("Total Columns:", df.shape[1])

# ============================================
# 5. DATASET INFORMATION
# ============================================

print("\n========== DATASET INFORMATION ==========")
df.info()

# ============================================
# 6. MISSING VALUES
# ============================================

print("\n========== MISSING VALUES ==========")

missing_values = df.isnull().sum()

print(missing_values)

print("\nTotal Missing Values:",
      missing_values.sum())

# ============================================
# 7. DUPLICATE RECORDS
# ============================================

print("\n========== DUPLICATE RECORDS ==========")

duplicates = df.duplicated().sum()

print("Total Duplicate Records:", duplicates)

# ============================================
# 8. STATISTICAL SUMMARY
# ============================================

print("\n========== STATISTICAL SUMMARY ==========")

print(df.describe())

# ============================================
# 9. TARGET CLASS DISTRIBUTION
# ============================================

print("\n========== CLASS DISTRIBUTION ==========")

print(df["Result"].value_counts())

print("\n========== CLASS PERCENTAGE ==========")

print(df["Result"].value_counts(normalize=True) * 100)

# ============================================
# 10. PHISHING AND LEGITIMATE COUNT GRAPH
# ============================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Result",
    hue="Result",
    legend=False
)

plt.title("Phishing vs Legitimate Websites")
plt.xlabel("-1 = Phishing, 1 = Legitimate")
plt.ylabel("Number of Websites")

plt.tight_layout()

plt.savefig("reports/class_distribution.png")

plt.show()

# ============================================
# 11. CORRELATION ANALYSIS
# ============================================

correlation = df.corr(numeric_only=True)

print("\n========== CORRELATION WITH RESULT ==========")

print(
    correlation["Result"].sort_values(
        ascending=False
    )
)

# ============================================
# 12. CORRELATION HEATMAP
# ============================================

plt.figure(figsize=(16, 12))

sns.heatmap(
    correlation,
    cmap="coolwarm",
    annot=False,
    linewidths=0.3
)

plt.title("Phishing Website Detection - Correlation Heatmap")

plt.tight_layout()

plt.savefig("reports/correlation_heatmap.png")

plt.show()

# ============================================
# 13. IMPORTANT FEATURE DISTRIBUTIONS
# ============================================

features = [
    "URL_Length",
    "SSLfinal_State",
    "web_traffic",
    "age_of_domain"
]

for feature in features:

    if feature in df.columns:

        plt.figure(figsize=(8, 5))

        sns.countplot(
            data=df,
            x=feature,
            hue="Result"
        )

        plt.title(f"{feature} vs Website Class")

        plt.xlabel(feature)

        plt.ylabel("Number of Websites")

        plt.tight_layout()

        plt.savefig(
            f"reports/{feature}_distribution.png"
        )

        plt.show()

# ============================================
# 14. SAVE DATASET ANALYSIS SUMMARY
# ============================================

with open(
    "reports/eda_summary.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write("PHISHING WEBSITE DETECTION - EDA REPORT\n")

    file.write("\nDataset Shape: " + str(df.shape))

    file.write("\n\nMissing Values:\n")
    file.write(str(missing_values))

    file.write("\n\nDuplicate Records: ")
    file.write(str(duplicates))

    file.write("\n\nClass Distribution:\n")
    file.write(str(df["Result"].value_counts()))

    file.write("\n\nStatistical Summary:\n")
    file.write(str(df.describe()))

print("\n===================================")
print("EDA COMPLETED SUCCESSFULLY!")
print("Graphs saved in reports folder.")
print("EDA summary saved successfully.")
print("===================================")
