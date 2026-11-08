"""Clean the dataset, drop unnecessary columns and create train / test splits."""
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "tourism_project/data/tourism.csv"
TARGET = "ProdTaken"

# Load directly from the repository data folder
df = pd.read_csv(DATA_PATH)
print("Raw shape:", df.shape)

# --- Cleaning -------------------------------------------------------------
# Unnecessary columns: unnamed index column + CustomerID (identifier, no signal)
drop_cols = [c for c in df.columns if c.startswith("Unnamed")] + ["CustomerID"]
df = df.drop(columns=drop_cols)

# Standardise inconsistent category labels
df["Gender"] = df["Gender"].str.strip().replace({"Fe Male": "Female"})
df["MaritalStatus"] = df["MaritalStatus"].str.strip().replace({"Unmarried": "Single"})
for c in df.select_dtypes(include="object").columns:
    df[c] = df[c].str.strip()

# Remove duplicates; fill any missing values (numeric -> median, categorical -> mode)
df = df.drop_duplicates()
for c in df.columns:
    if df[c].isna().any():
        df[c] = df[c].fillna(df[c].median() if df[c].dtype != "object" else df[c].mode()[0])

print("Clean shape:", df.shape)

# --- Split ---------------------------------------------------------------
X = df.drop(columns=[TARGET])
y = df[TARGET]
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Saved in the working directory; the workflow uploads these as an artifact
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print(f"Train: {Xtrain.shape}, Test: {Xtest.shape}")
print("Train target mean:", round(ytrain.mean(), 3), "| Test target mean:", round(ytest.mean(), 3))
print("Saved Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv")
