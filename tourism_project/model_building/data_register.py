"""Register and validate the tourism dataset stored in the repository."""
import sys
import pandas as pd

DATA_PATH = "tourism_project/data/tourism.csv"

EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier", "DurationOfPitch",
    "Occupation", "Gender", "NumberOfPersonVisiting", "NumberOfFollowups", "ProductPitched",
    "PreferredPropertyStar", "MaritalStatus", "NumberOfTrips", "Passport",
    "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting", "Designation",
    "MonthlyIncome",
]

df = pd.read_csv(DATA_PATH)

missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
if missing:
    print(f"VALIDATION FAILED - missing columns: {missing}")
    sys.exit(1)

extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]
print("VALIDATION PASSED - all expected columns are present.")
print(f"Extra / unnamed columns (removed later in data prep): {extra}")

print("\n--- Dataset summary ---")
print(f"Rows: {df.shape[0]}  |  Columns: {df.shape[1]}")
print(f"Duplicate rows: {df.duplicated().sum()}")
print("\nData types:")
print(df.dtypes.to_string())
print("\nMissing values per column:")
print(df.isna().sum().to_string())
print("\nTarget distribution (ProdTaken):")
print(df["ProdTaken"].value_counts(normalize=True).round(3).to_string())
print("\nDataset registered successfully.")
