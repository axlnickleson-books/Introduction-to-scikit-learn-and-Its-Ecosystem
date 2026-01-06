
print("=== Step 1: Import required libraries ===")
import pandas as pd

print("=== Step 2: Define online dataset URL and load with pandas ===")
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
print(f"Downloading dataset from: {url}")

df = pd.read_csv(url)
print("Dataset loaded successfully from online repository!")
print()

print("=== Step 3: Basic inspection ===")
print("First 5 rows of the dataset:")
print(df.head())
print()

print("Dataset shape (rows, columns):", df.shape)
print("Column names:")
print(df.columns.tolist())
print()
df.describe().to_csv("titanic_statistics.csv")
print("=== Step 4: Data types and non-null counts ===")
print("DataFrame info():")
print(df.info())
print()

print("=== Step 5: Summary statistics for numerical features ===")
print(df.describe())
print()

print("=== Step 6: Summary statistics for categorical features ===")
print(df.describe(include='object'))
print()
df.describe(include='object').to_csv("titanic_statistics_object.csv")
print("=== Step 7: Missing values per column ===")
print(df.isna().sum())
print()

print("=== Step 8: Example of value counts for a key column (Survived) ===")
if "Survived" in df.columns:
    print(df["Survived"].value_counts(dropna=False))
else:
    print("Column 'Survived' not found in this dataset.")
print()

print("=== Workflow Complete: Online dataset loaded and inspected ===")