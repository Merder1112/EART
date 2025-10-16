# AI_project_master/app.py
import streamlit as st
import pandas as pd
import joblib, json, time
from pathlib import Path

st.set_page_config(page_title="Earthquake Alert", page_icon="🌎", layout="wide")

# ---- Paths ----
BASE = Path("AI_project_master")
DATA_PATH   = BASE / "data" / "earthquakes.csv"
MODEL_PATH  = BASE / "models" / "earthquake_model.pkl"
ENC_PATH    = BASE / "models" / "label_encoder.pkl"
ANN_PATH    = BASE / "storage" / "public_announcements.json"
METRICS_CSV = BASE / "metrics" / "metrics.csv"

st.title("🌎 ระบบแจ้งเตือนแผ่นดินไหว")
st.caption("เลือกเหตุการณ์/กรอกค่า → ทำนายด้วย AI → เผยแพร่ประกาศ  •  (โมเดล: Decision Tree)")

# ---- Check required files ----
missing = [p for p in [DATA_PATH, MODEL_PATH, ENC_PATH] if not Path(p).exists()]
if missing:
    st.error("ไฟล์ต่อไปนี้ยังไม่พบ:\n" + "\n".join(f"- {p}" for p in missing))
    st.info("โปรดตรวจสอบว่าใส่ earthquakes.csv ถูกที่ และได้รัน train.py แล้ว")
    st.stop()

# ---- Load data/model/encoder ----
df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)
le    = joblib.load(ENC_PATH)

# Basic numeric cleaning for UI
for col in ["magnitude","depth","cdi","mmi","sig"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col].fillna(df[col].median(), inplace=True)

st.write(f"📦 ข้อมูลทั้งหมด: {len(df):,} แถว")
with st.expander("ดูตัวอย่างข้อมูล (5 แถว)"):
    st.dataframe(df.head(5), use_container_width=True)

# ---- Training Results (ตอบโจทย์ข้อ 4) ----
with st.expander("📊 Training Results (หลาย features/parameters)"):
    if METRICS_CSV.exists():
        mdf = pd.read_csv(METRICS_CSV)
        if not mdf.empty:
            st.dataframe(mdf, use_container_width=True)
            top = mdf.sort_values("accuracy", ascending=False).iloc[0]
            st.success(
                f"โมเดลที่เลือก: feature_set = {top['feature_set']}, "
                f"params = {top['params']}, accuracy = {top['accuracy']}"
            )
        else:
            st.info("metrics.csv ว่างเปล่า — โปรดรัน train.py เพื่อสร้างผลลัพธ์")
    else:
        st.info("ยังไม่พบไฟล์ metrics.csv — โปรดรัน train.py ก่อน")

# ---- Officer section ----
st.subheader("👮‍♀️ เลือกเหตุการณ์หรือกรอกค่าเอง แล้วกดทำนาย")
latest = df.tail(200).reset_index(drop=True)

left, right = st.columns([1,1])
with left:
    st.markdown("**เลือกเหตุการณ์ (200 แถวล่าสุด)**")
    idx = st.number_input("หมายเลขแถว:", min_value=0, max_value=len(latest)-1,
                          value=len(latest)-1, step=1)
    row = latest.iloc[int(idx)].to_dict()

with right:
    st.markdown("**กรอกค่าเอง (ทับค่าจากแถวที่เลือกได้)**")

    def defval(k, default):
        v = row.get(k, default)
        try:
            return float(v) if pd.notna(v) else float(default)
        except Exception:
            return float(default)

    mag = st.number_input("magnitude", value=defval("magnitude", 5.0))
    dep = st.number_input("depth",     value=defval("depth", 10.0))
    cdi = st.number_input("cdi",       value=defval("cdi", 3.0))
    mmi = st.number_input("mmi",       value=defval("mmi", 3.0))
    sig = st.number_input("sig",       value=defval("sig", 300.0))

    inputs = pd.DataFrame([{
        "magnitude": mag, "depth": dep, "cdi": cdi, "mmi": mmi, "sig": sig
    }])

if st.button("🧠 ทำนายด้วย AI", use_container_width=True):
    # ใช้เฉพาะคอลัมน์ที่มีจริงใน dataset/ตอนเทรน
    feat_cols = [c for c in ["magnitude","depth","cdi","mmi","sig"] if c in df.columns]
    X = inputs[feat_cols]
    y_id = model.predict(X)[0]
    y_label = le.inverse_transform([y_id])[0]
    st.success(f"ผลทำนายระดับแจ้งเตือน: **{str(y_label).upper()}**")
    st.session_state["last_pred"] = {
        "risk": str(y_label),
        "inputs": inputs.iloc[0].to_dict(),
        "region": str(row.get("place", "Affected area")) if "place" in row else "Affected area"
    }

# ---- Publish announcement ----
if "last_pred" in st.session_state:
    st.divider()
    st.subheader("📢 เผยแพร่ประกาศ (ประชาชนจะเห็นด้านล่าง)")
    pred = st.session_state["last_pred"]
    region = st.text_input("พื้นที่/ภูมิภาค", value=pred["region"])
    msg_default = f"ตรวจพบเหตุสั่นสะเทือนระดับ {pred['risk'].upper()} — โปรดปฏิบัติตามคำแนะนำความปลอดภัย"
    message = st.text_area("ข้อความประกาศ", value=msg_default, height=80)
    tips = ["อยู่ใต้โต๊ะ/โครงสร้างแข็งแรง", "หลีกเลี่ยงลิฟต์/กระจก", "ปิดแก๊ส/ไฟฟ้า"]

    if st.button("✅ เผยแพร่ประกาศ", type="primary", use_container_width=True):
        ANN_PATH.parent.mkdir(parents=True, exist_ok=True)
        doc = {
            "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "announcements": [{
                "id": str(int(time.time())),
                "region": region,
                "risk_level": pred["risk"],
                "message": message,
                "tips": tips,
                "inputs": pred["inputs"]
            }]
        }
        with open(ANN_PATH, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
        st.success("เผยแพร่แล้ว! เลื่อนลงไปดูพื้นที่ประกาศด้านล่างได้เลย 👇")

# ---- Announcements (public view) ----
st.divider()
st.subheader("🚨 พื้นที่ประกาศ (แสดงต่อประชาชน)")
if not ANN_PATH.exists():
    st.info("ยังไม่มีประกาศล่าสุด")
else:
    ann = json.load(open(ANN_PATH, encoding="utf-8"))
    st.caption(f"อัปเดตล่าสุด: {ann.get('last_updated','-')}")
    for a in ann.get("announcements", []):
        level = str(a.get("risk_level","")).lower()
        color = {"green":"🟢","yellow":"🟡","orange":"🟠","red":"🔴"}.get(level, "🔶")
        st.markdown(f"### {color} ระดับแจ้งเตือน: **{str(a.get('risk_level','')).upper()}**")
        st.write(f"พื้นที่: **{a.get('region','-')}**")
        if a.get("message"):
            st.write(a["message"])
        if a.get("tips"):
            st.write("คำแนะนำ:")
            for t in a["tips"]:
                st.write(f"- {t}")

# ---- Credits (ตอบข้อ 2 แสดงแหล่งที่มา) ----
with st.expander("ℹ️ Dataset Source / ข้อมูลอ้างอิง"):
    st.write("Dataset: Kaggle – Earthquake Alert Prediction (ผู้จัดทำ: Ahmed Uzaki) หรือข้อมูลจริงจากหน่วยงานที่เกี่ยวข้อง (โปรดยืนยันแหล่งที่มาในรายงานของคุณ)")
