print("=== Step 1: Import required libraries ===")
import pandas as pd
from sklearn.datasets import load_wine

print("=== Step 2: Load the dataset from sklearn ===")
data = load_wine()
print("Dataset loaded successfully!")
print("Dataset keys:", data.keys())
print()

print("=== Step 3: Extract features, target, and feature names ===")
X = data.data
y = data.target
feature_names = data.feature_names

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)
print("First 5 rows of X:\n", X[:5])
print("First 10 target values:", y[:10])
print()

print("=== Step 4: Convert NumPy arrays into a pandas DataFrame ===")
df = pd.DataFrame(X, columns=feature_names)
df["target"] = y
print("DataFrame created! First 5 rows:")
print(df.head())
print()

print("=== Step 5: Inspect DataFrame structure ===")
print("DataFrame info():")
print(df.info())
print()

print("=== Step 6: Display summary statistics ===")
print(df.describe())
print()

print("=== Workflow Complete: Data Loaded and Inspected ===")

df.describe().to_csv("Wine_statistics.csv")