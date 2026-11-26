
import pandas as pd
from scipy.io import arff

# Load original ARFF dataset
data, metadata = arff.loadarff(
    "dataset/Training Dataset.arff"
)

# Convert into DataFrame
df = pd.DataFrame(data)

# Convert byte columns if present
for column in df.select_dtypes(include=["object"]).columns:
    df[column] = df[column].apply(
        lambda x: x.decode("utf-8")
        if isinstance(x, bytes) else x
    )

# Save as CSV
df.to_csv(
    "dataset/phishing_websites.csv",
    index=False
)

print("Dataset converted successfully!")
print("Dataset shape:", df.shape)
print(df.head())
