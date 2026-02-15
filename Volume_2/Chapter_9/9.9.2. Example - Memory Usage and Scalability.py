import sys
import joblib
import numpy as np

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# ------------------------------------------------------------
# Example data (replace this with your own X, y)
# ------------------------------------------------------------
X, y = make_classification(
    n_samples=5000,
    n_features=50,
    n_informative=20,
    n_redundant=10,
    n_classes=2,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ------------------------------------------------------------
# Train models
# ------------------------------------------------------------
rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
).fit(X_train, y_train)

lgb = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1,
).fit(X_train, y_train)

# ------------------------------------------------------------
# Estimate serialized model sizes (MB)
# ------------------------------------------------------------
models = [("RF", rf), ("XGB", xgb), ("LGB", lgb)]

sizes_mb = {
    name: len(joblib.dumps(model)) / (1024**2)  # bytes -> MB
    for name, model in models
}

for name, size in sizes_mb.items():
    print(f"{name} model size: {size:.2f} MB")

# Optional: also save to disk and show file sizes (often what you really care about)
# ------------------------------------------------------------
for name, model in models:
    path = f"{name.lower()}_model.joblib"
    joblib.dump(model, path, compress=3)
    file_mb = (joblib.os.path.getsize(path)) / (1024**2)
    print(f"{name} saved file size (compress=3): {file_mb:.2f} MB -> {path}")