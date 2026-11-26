
import pandas as pd

# Step 1: Load dataset
df = pd.read_csv("dataset/phishing_websites.csv")

# Step 2: Display first 5 rows
print("\n========== FIRST 5 ROWS ==========")
print(df.head())

# Step 3: Display last 5 rows
print("\n========== LAST 5 ROWS ==========")
print(df.tail())

# Step 4: Dataset shape
print("\n========== DATASET SHAPE ==========")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# Step 5: Display column names
print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

# Step 6: Data types
print("\n========== DATA TYPES ==========")
print(df.dtypes)

# Step 7: Dataset information
print("\n========== DATASET INFORMATION ==========")
df.info()

# Step 8: Statistical summary
print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())

# Step 9: Target column information
print("\n========== TARGET COLUMN ==========")
print("Target column: Result")
print(df["Result"].value_counts())

print("\nDataset loaded and analysed successfully!")
