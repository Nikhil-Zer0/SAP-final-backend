import pandas as pd
import numpy as np

# Read the CSV
df = pd.read_csv('hiring_data.csv')

print("🔍 Dataset Validation Report\n")
print(f"Shape: {df.shape}\n")
print("First 5 rows:")
print(df.head(), "\n")

print("Column names:", df.columns.tolist())
print("Data types:\n", df.dtypes, "\n")

print("Missing values per column:")
print(df.isna().sum(), "\n")

print("Unique values per column:")
for col in df.columns:
    print(f"{col}: {df[col].unique().tolist()}")

print("\nEmptyEntries/Whitespace:")
for col in df.select_dtypes(include='object').columns:
    empty = df[col].str.strip().eq('').sum()
    whitespace = df[col].str.isspace().sum()
    if empty > 0 or whitespace > 0:
        print(f"  {col}: {empty} empty, {whitespace} whitespace")

# Check for trailing spaces
print("\nTrailing spaces in 'gender' column:")
gender_values = df['gender'].astype(str).tolist()
for val in set(gender_values):
    if val != val.strip():
        print(f"  '{val}' → should be '{val.strip()}'")

# Check hired values
print(f"\n'hired' column values: {df['hired'].unique()}")
if not set(df['hired'].unique()).issubset({0, 1}):
    print("⚠️  'hired' contains non-binary values!")