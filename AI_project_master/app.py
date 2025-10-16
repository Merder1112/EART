# app.py — หน้าเดียว: เจ้าหน้าที่ด้านบน / พื้นที่ประกาศด้านล่าง
import streamlit as st
import pandas as pd
import joblib, json, os, time
from pathlib import Path

st.set_page_config(page_title="Earthquake Alert", page_icon="🌎", layout="wide")

DATA_PATH  = "AI_project_master/data/earthquakes.csv"
MODEL_PATH = "AI_project_master/earthquake_model.pkl"
ENC_PATH   = "AI_project_master/label_encoder.pkl"
ANN_PATH   = "AI_project_master/storage/public_announcements.json"

st.title("🌎 ระบบแจ้งเตือนแผ่นดินไหว")
st.caption("เลือกเหตุการณ์/กรอกค่า → ทำนายด้วย AI → เผยแพร่ประกาศ ")

# ตรวจไฟล์จำเป็น
missing = [p for p in [DATA_PATH, MODEL_PATH, ENC_PATH] if not Path(p).exists()]
if missing:
    st.error("ไฟล์ต่อไปนี้ยังไม่พบ:\n" + "\n".join(f"- {p}" for p in missing))
    st.stop()

# โหลดข้อมูล/โมเดล
df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)
le    = joblib.load(ENC_PATH)

st.write(f"📦 ข้อมูลทั้งหมด: {len(df):,} แถว")
with st.expander("ดูตัวอย่างข้อมูล (5 แถว)"):
    st.dataframe(df.head(5), use_container_width=True)

# ส่วนเจ้าหน้าที่
st.subheader("👮‍♀️ เลือกเหตุการณ์หรือกรอกค่าเอง แล้วกดทำนาย")
latest = df.tail(200).reset_index(drop=True)

left, right = st.columns([1,1])
with left:
    st.markdown("**เลือกเหตุการณ์ (200 แถวล่าสุด)**")
    idx = st.number_input("หมายเลขแถว:", min_value=0, max_value=len(latest)-1, value=len(latest)-1, step=1)
    row = latest.iloc[int(idx)].to_dict()

with right:
    st.markdown("**กรอกค่าเอง (ทับค่าจากแถวที่เลือกได้)**")
    mag  = st.number_input("magnitude", value=float(row.get("magnitude", 5.0)))
    dep  = st.number_input("depth",     value=float(row.get("depth", 10.0)))
    cdi  = st.number_input("cdi",       value=float(row.get("cdi", 3.0)))
    mmi  = st.number_input("mmi",       value=float(row.get("mmi", 3.0)))
    sig  = st.number_input("sig",       value=float(row.get("sig", 300.0)))
    inputs = pd.DataFrame([{
        "magnitude": mag, "depth": dep, "cdi": cdi, "mmi": mmi, "sig": sig
    }])

if st.button("🧠 ทำนายด้วย AI", use_container_width=True):
    X = inputs[["magnitude","depth","cdi","mmi","sig"]]
    y_id = model.predict(X)[0]
    y_label = le.inverse_transform([y_id])[0]
    st.success(f"ผลทำนายระดับแจ้งเตือน: **{y_label.upper()}**")
    st.session_state["last_pred"] = {
        "risk": y_label,
        "inputs": inputs.iloc[0].to_dict(),
        "region": str(row.get("place","Affected area"))  # ถ้ามีคอลัมน์ place จะเอามาแสดง
    }

# เผยแพร่ประกาศ → เขียนลงไฟล์ JSON
if "last_pred" in st.session_state:
    st.divider()
    st.subheader("📢 เผยแพร่ประกาศ (ประชาชนจะเห็นด้านล่าง)")
    pred = st.session_state["last_pred"]
    region = st.text_input("พื้นที่/ภูมิภาค", value=pred["region"])
    msg_default = f"ตรวจพบเหตุสั่นสะเทือนระดับ {pred['risk'].upper()} — โปรดปฏิบัติตามคำแนะนำความปลอดภัย"
    message = st.text_area("ข้อความประกาศ", value=msg_default, height=80)
    tips = ["อยู่ใต้โต๊ะ/โครงสร้างแข็งแรง", "หลีกเลี่ยงลิฟต์/กระจก", "ปิดแก๊ส/ไฟฟ้า"]

    if st.button("✅ เผยแพร่ประกาศ", type="primary", use_container_width=True):
        os.makedirs("storage", exist_ok=True)
        doc = {"last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "announcements": [{
                   "id": str(int(time.time())),
                   "region": region,
                   "risk_level": pred["risk"],
                   "message": message,
                   "tips": tips,
                   "inputs": pred["inputs"]
               }]}
        with open(ANN_PATH, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
        st.success("เผยแพร่แล้ว! เลื่อนลงไปดูพื้นที่ประกาศด้านล่างได้เลย 👇")

# พื้นที่ประกาศ (ประชาชนเห็น)
st.divider()
st.subheader("🚨 พื้นที่ประกาศ (แสดงต่อประชาชน)")
if not Path(ANN_PATH).exists():
    st.info("ยังไม่มีประกาศล่าสุด")
else:
    ann = json.load(open(ANN_PATH, encoding="utf-8"))
    st.caption(f"อัปเดตล่าสุด: {ann['last_updated']}")
    for a in ann["announcements"]:
        color = {"green":"🟢","yellow":"🟡","orange":"🟠","red":"🔴"}.get(a["risk_level"], "🔶")
        st.markdown(f"### {color} ระดับแจ้งเตือน: **{a['risk_level'].upper()}**")
        st.write(f"พื้นที่: **{a.get('region','-')}**")
        st.write(a["message"])
        st.write("คำแนะนำ:")
        for t in a.get("tips", []):
            st.write(f"- {t}")

