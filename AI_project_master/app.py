# AI_project_master/app.py
import os, json, time
from pathlib import Path

import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Earthquake Alert", page_icon="🌎", layout="wide")

# ---------------- Paths ----------------
BASE        = Path(__file__).resolve().parent
DATA_PATH   = BASE / "data" / "earthquakes.csv"
MODEL_PATH  = BASE / "models" / "earthquake_model.pkl"
ENC_PATH    = BASE / "models" / "label_encoder.pkl"
ANN_PATH    = BASE / "storage" / "public_announcements.json"
METRICS_CSV = BASE / "metrics" / "metrics.csv"
FIG_CM      = BASE / "metrics" / "figs" / "confusion_matrix.png"

# ---------------- Sidebar: Mode & Admin PIN ----------------
st.sidebar.title("🌎 Earthquake Alert")
mode = st.sidebar.radio("เลือกโหมด", ["Public Alert", "Admin Dashboard"], index=0)

# (ไม่ปลอดภัยระดับ production แต่เพียงพอสำหรับงานส่ง)
DEFAULT_PIN = os.environ.get("EA_ADMIN_PIN", "cis2025")
admin_ok = False
if mode == "Admin Dashboard":
    pin = st.sidebar.text_input("Admin PIN", type="password", help="ค่าเริ่มต้น: cis2025 (เปลี่ยนได้ด้วย ENV EA_ADMIN_PIN)")
    admin_ok = (pin == DEFAULT_PIN)

# ---------------- Header ----------------
st.title("🌎 ระบบแจ้งเตือนแผ่นดินไหว")
if mode == "Public Alert":
    st.caption("เลือกเหตุการณ์/กรอกค่า → ทำนายด้วย AI (Decision Tree) → เผยแพร่ประกาศ")
else:
    st.caption("แดชบอร์ดสำหรับเจ้าหน้าที่: ดูความแม่นยำ/พารามิเตอร์/ฟีเจอร์สำคัญ/เมทริกซ์ความสับสน")

# ---------------- Common: Check files ----------------
missing = [p for p in [DATA_PATH, MODEL_PATH, ENC_PATH] if not p.exists()]
if mode == "Public Alert":
    if missing:
        st.error("ไม่พบไฟล์ต่อไปนี้:\n" + "\n".join(f"- {p}" for p in missing))
        st.stop()
else:
    # ในหน้า Admin ให้ยังแสดงแดชบอร์ดเท่าที่มี (ถ้าบางไฟล์ยังไม่พร้อม จะแจ้งเตือนเฉพาะส่วน)
    pass

# ---------------- Load (if exists) ----------------
df = pd.read_csv(DATA_PATH) if DATA_PATH.exists() else None
model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
le    = joblib.load(ENC_PATH)   if ENC_PATH.exists()   else None

# ========== MODE: ADMIN DASHBOARD ==========
if mode == "Admin Dashboard":
    if not admin_ok:
        st.warning("โปรดกรอก Admin PIN ทางแถบซ้ายเพื่อเข้าถึงแดชบอร์ด")
        st.stop()

    colA, colB, colC = st.columns([1,1,1])

    # 1) Accuracy & Best Params from metrics.csv
    with colA:
        st.subheader("📊 Model Accuracy")
        if METRICS_CSV.exists():
            mdf = pd.read_csv(METRICS_CSV)
            if not mdf.empty and "accuracy" in mdf.columns:
                top = mdf.sort_values("accuracy", ascending=False).iloc[0]
                acc = float(top["accuracy"])
                st.metric("Accuracy (Test)", f"{acc*100:.2f}%")
            else:
                st.info("metrics.csv ว่างหรือไม่มีคอลัมน์ accuracy")
        else:
            st.info("ยังไม่พบ metrics.csv (สร้างจากโน้ตบุ๊กหรือ train.py)")

    with colB:
        st.subheader("⚙️ Best Parameters")
        if METRICS_CSV.exists():
            mdf = pd.read_csv(METRICS_CSV)
            if not mdf.empty:
                top = mdf.sort_values("accuracy", ascending=False).iloc[0]
                st.code(f"feature_set: {top.get('feature_set','-')}\n"
                        f"features   : {top.get('features','-')}\n"
                        f"params     : {top.get('params','-')}", language="text")
            else:
                st.info("ยังไม่มีข้อมูล params ใน metrics.csv")
        else:
            st.info("ยังไม่พบ metrics.csv")

    with colC:
        st.subheader("🕒 Model Files")
        rows = []
        for p in [MODEL_PATH, ENC_PATH, METRICS_CSV]:
            if p.exists():
                ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime))
                rows.append({"file": str(p.name), "status": "OK", "updated": ts})
            else:
                rows.append({"file": str(p.name), "status": "missing", "updated": "-"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.divider()

    # 2) Feature Importances (from DecisionTree)
    st.subheader("🌳 Feature Importances (Decision Tree)")
    if model is not None and hasattr(model, "feature_importances_") and df is not None:
        # ใช้ฟีเจอร์ที่เว็บรองรับ
        feat_cols = [c for c in ["magnitude","depth","cdi","mmi","sig"] if c in df.columns]
        try:
            imps = pd.Series(model.feature_importances_, index=feat_cols).sort_values(ascending=False)
            st.bar_chart(imps)  # chart สำหรับ admin เท่านั้น
            st.caption("หมายเหตุ: ค่า importance สูง = โมเดลให้ความสำคัญกับฟีเจอร์นั้นมากในการตัดสินใจ")
        except Exception as e:
            st.error(f"ไม่สามารถอ่าน feature_importances_: {e}")
    else:
        st.info("ยังไม่พบโมเดลหรือตัวชี้วัดความสำคัญของฟีเจอร์ (ต้องใช้ Decision Tree)")

    # 3) Confusion Matrix (ถ้ามีรูปจาก Jupyter)
    st.subheader("🧩 Confusion Matrix (จากการประเมินในโน้ตบุ๊ก)")
    if FIG_CM.exists():
        st.image(str(FIG_CM), caption="Confusion Matrix (ไฟล์จาก notebooks)")
    else:
        st.info("ยังไม่พบไฟล์รูป Confusion Matrix (metrics/figs/confusion_matrix.png)")

    st.divider()
    st.subheader("📝 หมายเหตุสำหรับเจ้าหน้าที่")
    st.write(
        "- Accuracy เป็นผลจากชุดทดสอบ/การทดลอง (อ้างอิง metrics.csv)\n"
        "- แนะนำติดตาม Precision/Recall รายคลาสในโน้ตบุ๊กเพื่อดูจุดสับสน\n"
        "- ปรับปรุงโมเดล: อัปเดตรายการฟีเจอร์/พารามิเตอร์ใน train.py หรือ notebooks แล้วสร้างไฟล์ใหม่\n"
    )

# ========== MODE: PUBLIC ALERT ==========
if mode == "Public Alert":
    # ทำความสะอาดตัวเลขสำหรับ UI
    for col in ["magnitude","depth","cdi","mmi","sig"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    st.write(f"📦 ข้อมูลทั้งหมด: {len(df):,} แถว")
    with st.expander("ดูตัวอย่างข้อมูล (5 แถว)"):
        st.dataframe(df.head(5), use_container_width=True)

    # Officer panel
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

    # Publish announcement
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

    # Public area
    st.divider()
    st.subheader("🚨 พื้นที่ประกาศ (แสดงต่อประชาชน)")
    if ANN_PATH.exists():
        ann = json.load(open(ANN_PATH, encoding="utf-8"))
        st.caption(f"อัปเดตล่าสุด: {ann.get('last_updated','-')}")
        for a in ann.get("announcements", []):
            level = str(a.get("risk_level","")).lower()
            color = {"green":"🟢","yellow":"🟡","orange":"🟠","red":"🔴"}.get(level, "🔶")
            st.markdown(f"### {color} ระดับแจ้งเตือน: **{str(a.get('risk_level','')).upper()}**")
            st.write(f"พื้นที่: **{a.get('region','-')}**")
            if a.get("message"): st.write(a["message"])
            if a.get("tips"):
                st.write("คำแนะนำ:")
                for t in a["tips"]:
                    st.write(f"- {t}")
    else:
        st.info("ยังไม่มีประกาศล่าสุด")

# Credits (ไว้ตามข้อ 2)
with st.expander("ℹ️ Dataset Source / ข้อมูลอ้างอิง"):
    st.write("Dataset: Kaggle – Earthquake Alert Prediction (Ahmed Uzaki) หรือข้อมูลจริงจากหน่วยงานที่เกี่ยวข้อง")
