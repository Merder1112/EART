# AI_project_master/train.py
import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

# ---- Paths ----
BASE = Path("AI_project_master")
DATA_PATH  = BASE / "data" / "earthquakes.csv"
MODELS_DIR = BASE / "models"
METRICS_DIR= BASE / "metrics"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

# ---- Load data ----
df = pd.read_csv(DATA_PATH)

# numeric columns we may use
num_cols_all = ["magnitude", "depth", "cdi", "mmi", "sig"]
num_cols = [c for c in num_cols_all if c in df.columns]

# coerce to numeric & fillna
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
    df[c].fillna(df[c].median(), inplace=True)

# target
if "alert" not in df.columns:
    raise ValueError("Dataset ต้องมีคอลัมน์ 'alert' เป็นเป้าหมาย (ระดับแจ้งเตือน)")

le = LabelEncoder()
y = le.fit_transform(df["alert"].astype(str))

# Feature sets (เลือกเฉพาะที่มีจริง)
feature_sets = {
    "F1_basic": [c for c in ["magnitude", "depth", "sig"] if c in num_cols],
    "F2_all"  : [c for c in ["magnitude", "depth", "cdi", "mmi", "sig"] if c in num_cols],
}

# Param grid (ทดลองหลายพารามิเตอร์)
param_grid = [
    {"max_depth": d, "min_samples_split": mss}
    for d in [None, 5, 10, 20]
    for mss in [2, 5, 10]
]

rows = []
best = {"acc": -1, "model": None, "features": None, "params": None}

for fname, feats in feature_sets.items():
    if not feats:
        continue
    X = df[feats]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    for params in param_grid:
        clf = DecisionTreeClassifier(random_state=42, **params)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        rows.append({
            "feature_set": fname,
            "features": ",".join(feats),
            "params": json.dumps(params),
            "accuracy": round(acc, 4),
        })

        if acc > best["acc"]:
            best = {"acc": acc, "model": clf, "features": feats, "params": params}

# Save metrics table
metrics_df = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
metrics_df.to_csv(METRICS_DIR / "metrics.csv", index=False)

# Save best model + encoder
joblib.dump(best["model"], MODELS_DIR / "earthquake_model.pkl")
joblib.dump(le, MODELS_DIR / "label_encoder.pkl")

print("✅ Best accuracy:", round(best["acc"], 4))
print("🏷  Features:", best["features"])
print("⚙️  Params:", best["params"])
print("💾 Saved model:", MODELS_DIR / "earthquake_model.pkl")
print("💾 Saved encoder:", MODELS_DIR / "label_encoder.pkl")
print("📊 Metrics:", METRICS_DIR / "metrics.csv")
