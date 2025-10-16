# train.py
import os, json
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

BASE = Path("AI_project_master")
DATA_PATH  = BASE / "data" / "earthquakes.csv"
MODELS_DIR = BASE / "models"
METRICS_DIR= BASE / "metrics"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

# 1) โหลดข้อมูล
df = pd.read_csv(DATA_PATH)

# จัดการค่าว่างเบื้องต้น
for col in ["magnitude","depth","cdi","mmi","sig"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
df.fillna({c: df[c].median() for c in ["magnitude","depth","cdi","mmi","sig"] if c in df.columns}, inplace=True)

# 2) target: 'alert' -> encode
if "alert" not in df.columns:
    raise ValueError("Dataset ต้องมีคอลัมน์ 'alert' เป็นเป้าหมาย")
le = LabelEncoder()
y = le.fit_transform(df["alert"].astype(str))

# 3) กำหนด candidate features หลายชุด
feature_sets = {
    "F1_basic": ["magnitude","depth","sig"],
    "F2_all"  : ["magnitude","depth","cdi","mmi","sig"],
}
# กรองเฉพาะฟีเจอร์ที่มีจริง
feature_sets = {k:[c for c in v if c in df.columns] for k,v in feature_sets.items()}

# 4) กำหนดชุดพารามิเตอร์ที่จะทดลอง
param_grid = [
    {"max_depth": d, "min_samples_split": mss}
    for d in [None, 5, 10, 20]
    for mss in [2, 5, 10]
]

rows = []
best = {"acc": -1, "model": None, "features": None, "params": None}

for fname, feats in feature_sets.items():
    if len(feats) == 0: 
        continue
    X = df[feats]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    for params in param_grid:
        clf = DecisionTreeClassifier(random_state=42, **params)
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))

        rows.append({
            "feature_set": fname,
            "features": ",".join(feats),
            "params": json.dumps(params),
            "accuracy": round(acc, 4)
        })

        if acc > best["acc"]:
            best = {"acc": acc, "model": clf, "features": feats, "params": params}

# 5) บันทึกผลการทดลองทั้งหมด
metrics_df = pd.DataFrame(rows).sort_values(["accuracy"], ascending=False)
metrics_df.to_csv(METRICS_DIR / "metrics.csv", index=False)

# 6) เซฟโมเดลที่ดีที่สุด + encoder
joblib.dump(best["model"], MODELS_DIR / "earthquake_model.pkl")
joblib.dump(le, MODELS_DIR / "label_encoder.pkl")

print("✅ Best accuracy:", round(best["acc"], 4))
print("🏷  Features:", best["features"])
print("⚙️  Params:", best["params"])
print("💾 Saved:", MODELS_DIR / "earthquake_model.pkl", ",", MODELS_DIR / "label_encoder.pkl")
print("📊 Metrics:", METRICS_DIR / "metrics.csv")
