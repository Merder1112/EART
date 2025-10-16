# AI_project_master/app.py
import streamlit as st
import pandas as pd
import numpy as np  # NEW: ใช้คำนวณความมั่นใจ
import joblib, json, time
from pathlib import Path

st.set_page_config(page_title="Earthquake Alert", page_icon="🌎", layout="wide")

# --- Paths ---
BASE = Path(__file__).resolve().parent
DATA_PATH    = BASE / "data" / "earthquakes.csv"
MODEL_PATH   = BASE / "models" / "earthquake_model.pkl"
ENC_PATH     = BASE / "models" / "label_encoder.pkl"
ANN_PATH     = BASE / "storage" / "public_announcements.json"
METRICS_PATH = BASE / "metrics" / "metrics.csv"   # NEW: ตารางผลการทดลอง

st.title("🌎 ระบบแจ้งเตือนแผ่นดินไหว")
st.caption("เลือกเหตุการณ์กรอกค่าพารามิเตอร์ → ทำนายด้วย AI (Decision Tree) → เผยแพร่ประกาศ")

# --- Check required files ---
missing = [p for p in [DATA_PATH, MODEL_PATH, ENC_PATH] if not p.exists()]
if missing:
    st.error("ไม่พบไฟล์ต่อไปนี้:\n" + "\n".join(f"- {p}" for p in missing))
    st.stop()

# --- Load data/model/encoder ---
df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)
le    = joblib.load(ENC_PATH)

# --- TRY load overall accuracy from metrics.csv (รองรับทั้งรูปแบบ train.py และ notebook) ---
model_acc = None
cv_mean   = None
if METRICS_PATH.exists():
    try:
        mdf = pd.read_csv(METRICS_PATH)
        # รูปแบบจาก notebook: มี test_accuracy / cv_mean
        if "test_accuracy" in mdf.columns:
            row = mdf.sort_values(["test_accuracy","cv_mean"], ascending=False).iloc[0]
            model_acc = float(row["test_accuracy"])
            if "cv_mean" in mdf.columns:
                cv_mean = float(row["cv_mean"])
        # รูปแบบจาก train.py: มี accuracy อย่างเดียว
        elif "accuracy" in mdf.columns:
            row = mdf.sort_values(["accuracy"], ascending=False).iloc[0]
            model_acc = float(row["accuracy"])
    except Exception:
        pass

# Clean numeric for UI
for col in ["magnitude","depth","cdi","mmi","sig"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

st.write(f"ข้อมูลทั้งหมด : {len(df):,} แถว")
with st.expander("ดูตัวอย่างข้อมูล (10 แถว)"):
    st.dataframe(df.head(10), use_container_width=True)

# --- Officer section: เลือก/กรอก + ทำนาย ---
st.subheader("👮‍♀️ เจ้าหน้าที่ ศูนย์เตือนภัยพิบัติ")
st.caption("เลือกเหตุการณ์หรือกรอกค่าเอง แล้วกดทำนาย")

# NEW: แสดงความแม่นยำโดยรวมของโมเดล (จากการทดสอบ/ทดลอง)
if model_acc is not None:
    cols_info = st.columns(2)
    with cols_info[0]:
        st.metric("ความแม่นยำโดยรวมของโมเดล", f"{model_acc*100:.2f}%")
    if cv_mean is not None:
        with cols_info[1]:
            st.metric("Cross-Validation เฉลี่ย (5-fold)", f"{cv_mean*100:.2f}%")
else:
    st.info("ยังไม่พบไฟล์ metrics.csv สำหรับสรุปความแม่นยำโดยรวม (รันเทรน/โน้ตบุ๊กเพื่อสร้างไฟล์นี้ได้)")

latest = df.tail(200).reset_index(drop=True)

left, right = st.columns([1,1])
with left:
    st.markdown("**เลือกเหตุการณ์ ( 200 แถวล่าสุดข้อมูล )**")
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

    mag = st.number_input("magnitude ", value=defval("magnitude", 5.0))
    dep = st.number_input("depth",     value=defval("depth", 10.0))
    cdi = st.number_input("cdi",       value=defval("cdi", 3.0))
    mmi = st.number_input("mmi",       value=defval("mmi", 3.0))
    sig = st.number_input("sig",       value=defval("sig", 300.0))

    inputs = pd.DataFrame([{
        "magnitude": mag, "depth": dep, "cdi": cdi, "mmi": mmi, "sig": sig
    }])

if st.button(" ทำนายด้วย AI", use_container_width=True):
    feat_cols = [c for c in ["magnitude","depth","cdi","mmi","sig"] if c in df.columns]
    X = inputs[feat_cols]
    y_id = model.predict(X)[0]
    y_label = le.inverse_transform([y_id])[0]

    st.success(f"ผลทำนายระดับแจ้งเตือน : **{str(y_label).upper()}**")

    # NEW: แสดงความมั่นใจรายเหตุการณ์ (Predict Proba)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]               # array เช่น [0.05, 0.12, 0.20, 0.63]
        conf  = float(np.max(proba))                    # ค่ามั่นใจสูงสุดของคลาสที่ทำนาย
        # ชื่อคลาสตาม encoder
        labels = le.inverse_transform(np.arange(len(proba)))
        st.caption(f"ความมั่นใจของโมเดลต่อเหตุการณ์นี้: **{conf*100:.1f}%**")
        st.bar_chart(pd.DataFrame({"probability": proba}, index=[l.upper() for l in labels]))

    st.session_state["last_pred"] = {
        "risk": str(y_label),
        "inputs": inputs.iloc[0].to_dict(),
        "region": str(row.get("place", "Affected area")) if "place" in row else "Affected area"
    }

# --- Publish announcement ---
if "last_pred" in st.session_state:
    st.divider()
    st.subheader(" เผยแพร่ประกาศการแจ้งเตือนภัย")
    pred = st.session_state["last_pred"]
    region = st.text_input("พื้นที่/ภูมิภาค", value=pred["region"])
    msg_default = f"ตรวจพบเหตุสั่นสะเทือนระดับ {pred['risk'].upper()} — โปรดปฏิบัติตามคำแนะนำความปลอดภัย"
    message = st.text_area("ข้อความประกาศ", value=msg_default, height=80)
    tips = ["อยู่ใต้โต๊ะ/โครงสร้างแข็งแรง", "หลีกเลี่ยงลิฟต์/กระจก", "ปิดแก๊ส/ไฟฟ้า"]

    if st.button("ประกาศแจ้งเตือน", type="primary", use_container_width=True):
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
        st.success("เผยแพร่แล้ว ตัวอย่างการประกาศอยู่ด้านล่าง ")

# --- Public area ---
st.divider()
st.subheader("แสดงการแจ้งเตือนให้ประชาชน")
if ANN_PATH.exists():
    ann = json.load(open(ANN_PATH, encoding="utf-8"))
    for a in ann.get("announcements", []):
        level = str(a.get("risk_level","")).lower()
        color = {"green":"🟢","yellow":"🟡","orange":"🟠","red":"🔴"}.get(level, "🔶")
        st.markdown(f"### {color} ระดับแจ้งเตือน: **{str(a.get('risk_level','')).upper()}**")
        st.write(f"พื้นที่ : **{a.get('region','-')}**")
        if a.get("message"): st.write(a["message"])
        if a.get("tips"):
            st.write("คำแนะนำความปลอดภัย ดังนี้")
            for t in a["tips"]:
                st.write(f"- {t}")
else:
    st.info("ยังไม่มีประกาศล่าสุด")
