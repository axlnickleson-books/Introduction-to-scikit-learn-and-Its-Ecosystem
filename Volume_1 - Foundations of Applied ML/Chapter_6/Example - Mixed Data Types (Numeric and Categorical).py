import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Step 1: Fetch a mixed-type dataset from OpenML
data = fetch_openml("adult", version=2, as_frame=True)

# Features (X) and target (y) are taken directly from the Bunch
X = data.data          # feature matrix (DataFrame)
y = data.target        # target Series

print("Dataset name:", data.details.get("name", "adult"))
print("X shape:", X.shape)
print("y shape:", y.shape)

# Step 2: Identify numeric and categorical columns automatically
numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

print("Number of numeric features:", len(numeric_features))
print("Number of categorical features:", len(categorical_features))

# Step 3: Define preprocessing pipelines for each type
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Step 4: ColumnTransformer with explicit remainder handling
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ],
    remainder="drop"  # Drop any columns not explicitly listed
)

print("Example numeric columns:", numeric_features[:5])
print("Example categorical columns:", categorical_features[:5])

# Step 5: Build the complete pipeline
clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        random_state=42,
        n_estimators=100
    ))
])

# Step 6: Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
print("Training shape:", X_train.shape)
print("Testing  shape:", X_test.shape)

# Step 7: Fit the pipeline
clf.fit(X_train, y_train)

# Step 8: Evaluate performance
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)