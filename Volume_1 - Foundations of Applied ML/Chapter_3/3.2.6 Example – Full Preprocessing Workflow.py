print("=== Step 1: Import required libraries ===")
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

print("=== Step 2: Load Titanic dataset from GitHub ===")
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
print("Dataset loaded successfully!")
print("Shape of raw DataFrame:", df.shape)
print("First 5 rows:")
print(df.head())
print()

print("=== Step 3: Select features and target ===")
# Target column
target_col = "Survived"

# Feature columns (a mix of numeric and categorical)
feature_cols = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

X = df[feature_cols].copy()
y = df[target_col].copy()

print("Selected feature columns:", feature_cols)
print("Target column:", target_col)
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)
print()

print("=== Step 4: Train-test split ===")
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

print("Training set shape X_train:", X_train.shape)
print("Test set shape X_test:", X_test.shape)
print("Training target distribution:")
print(y_train.value_counts())
print("Test target distribution:")
print(y_test.value_counts())
print()

print("=== Step 5: Define numeric and categorical feature lists ===")
numeric_features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
categorical_features = ["Sex", "Embarked"]

print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)
print()

print("=== Step 6: Define preprocessing pipelines ===")
# Numeric: impute missing values with median, then scale
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical: impute most frequent category, then one-hot encode
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Combine into a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

print("Preprocessing pipelines defined:")
print(preprocessor)
print()

print("=== Step 7: Fit preprocessor on training data and transform ===")
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("Type of X_train_processed:", type(X_train_processed))
print("Type of X_test_processed:", type(X_test_processed))
print("Shape of X_train_processed:", X_train_processed.shape)
print("Shape of X_test_processed:", X_test_processed.shape)
print()

print("=== Step 8: (Optional) Inspect processed feature names ===")
# Get feature names after one-hot encoding (requires sklearn >= 1.0)
try:
    feature_names_num = numeric_features
    onehot = preprocessor.named_transformers_["cat"]["onehot"]
    feature_names_cat = onehot.get_feature_names_out(categorical_features)
    all_feature_names = np.concatenate([feature_names_num, feature_names_cat])
    print("Number of output features:", len(all_feature_names))
    print("First 20 feature names:")
    print(all_feature_names[:20])
except Exception as e:
    print("Could not extract feature names from preprocessor.")
    print("Reason:", e)
print()

print("=== Step 9: Final summary ===")
print("X_train_processed is ready for model training.")
print("X_test_processed is ready for model evaluation.")
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)