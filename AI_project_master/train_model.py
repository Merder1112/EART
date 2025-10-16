import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1) โหลดข้อมูล
df = pd.read_csv("data/earthquakes.csv")

# แปลง alert เป็นตัวเลข
le = LabelEncoder()
df["alert"] = le.fit_transform(df["alert"])

# 2) ฟีเจอร์และเป้าหมาย
X = df[["magnitude", "depth", "cdi", "mmi", "sig"]]
y = df["alert"]

# 3) แบ่งข้อมูล
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4) เทรนโมเดล
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5) ประเมินผล
y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# 6) บันทึกโมเดล
joblib.dump(model, "earthquake_model.pkl")
joblib.dump(le, "label_encoder.pkl")
print("\n💾 Saved: earthquake_model.pkl, label_encoder.pkl")
