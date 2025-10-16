# AI_project_master/app.py
import streamlit as st
import pandas as pd
import joblib, json, time, traceback
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Earthquake Alert", page_icon="🌎", layout="wide")

# --- Robust paths: based on this file location ---
BASE = Path(__file__).resolve().parent
DATA_PATH   = BASE / "data" / "earthquakes.csv"
MODELS_DIR  = BASE / "models"
MODEL_PATH  = MODELS_DIR / "earthquake_model.pkl"
ENC_PATH    = MODELS_DIR / "label_encoder.pkl"
STORE_DIR   = BASE / "storage"
ANN_PATH    = STORE_DIR / "public_announcements.json"
METRICS_DIR = BASE / "metrics"
METRICS_CSV = METRICS_DIR / "metrics.csv"

st.title("🌎 ระบบแจ้งเตือนแผ่นดินไหว")
st.caption("เลือกเหตุการณ์/กรอกค่า → ทำนายด้วย AI (Decision Tree) → เผยแพร่ประกาศ  •  แสดงผลการฝึกหลายชุดตามโจทย์")

# ---------- Helper ----------
def safe_read_csv(p: Path):
    try:
        return pd.read_csv(p)
    except Exception as e:
        st.error(f"อ่านไฟล์ CSV ไม่สำเร็จ: {p}\n\n{e}")
        st.code(traceback.format_exc())
        return None

def quick_train_decision_tree(df: pd.DataFrame):
    """เทรน Decision Tree อย่างเร็วในหน้าเว็บให้ผ่านโจทย์ (ข้อ 3–4) + สร้าง metrics.csv"""
    num_cols_all = ["magnitude", "depth", "cdi", "mmi", "sig"]
    have_cols = [c for c in num_cols_all if c in df.columns]
    if "alert" not in df.columns:
        st.error("ไม่พบคอลัมน์ 'alert' ใน dataset — ต้องมีคอลัมน์นี้เป็นเป้าหมาย")
        return False

    # numeric cleanup
    for c in have_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c].fillna(df[c].median(), inplace=True)

    le = LabelEncoder()
    y = le.fit_transform(df["alert"].astype(str))

    feature_sets = {
        "F1_basic": [c for c in ["magnitude","depth","sig"] if c in have_cols],
        "F2_all"  : [c for c in ["magnitude","depth","cdi","mmi","sig"] if c in have_cols],
    }
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
            acc = accuracy_score(y_test, clf.predict(X_test))
            rows.append({
                "feature_set": fname,
                "features": ",".join(feats),
                "params": json.dumps(params),
                "accuracy": round(acc, 4),
            })
            if acc > best["acc"]:
                best = {"acc": acc, "model": clf, "features": feats, "params": params}

    # save outputs
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("accuracy", ascending=False).to_csv(METRICS_CSV, index=False)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["model"], MODEL_PATH)
    joblib.dump(le, ENC_PATH)
    st.success(f"เทรนสำเร็จ! Best accuracy={round(best['acc'],4)} | "
               f"features={best['features']} | params={best['params']}")
    return True

# ---------- Section: Diagnostics (ไม่หยุดหน้าเว็บ) ----------
with st.expander("🧰 ตัวช่วยแก้ปัญหา (Diagnostics)"):
    st.write("ตำแหน่งไฟล์ที่ระบบอ้างอิง:")
    st.code(f"""
BASE        = {BASE}
DATA_PATH   = {DATA_PATH}
MODEL_PATH  = {MODEL_PATH}
ENC_PATH    = {ENC_PATH}
METRICS_CSV = {METRICS_CSV}
ANN_PATH    = {ANN_PATH}
""", language="text")

    exists = {
        "earthquakes.csv": DATA_PATH.exists(),
        "earthquake_model.pkl": MODEL_PATH.exists(),
        "label_encoder.pkl": ENC_PATH.exists(),
        "metrics.csv": METRICS_CSV.exists(),
    }
    st.write("สถานะไฟล์สำคัญ:")
    st.json(exists)

# ---------- Load dataset (หรือแจ้งปัญหา) ----------
df = None
if not DATA_PATH.exists():
    st.error("ไม่พบไฟล์ dataset: data/earthquakes.csv")
    st.info("โปรดวางไฟล์ Kaggle ไว้ที่: AI_project_master/data/earthquakes.csv แล้วกด Refresh")
else:
    df = safe_read_csv(DATA_PATH)

# ---------- Quick Train button (แก้เคสยังไม่มีโมเดล/metrics) ----------
if df is not None and (not MODEL_PATH.exists() or not ENC_PATH.exists() or not METRICS_CSV.exists()):
    st.warning("ยังไม่มีโมเดล/encoder หรือ metrics.csv — คุณสามารถเทรนอย่างรวดเร็วจากหน้านี้ได้")
    if st.button("🚀 Quick Train (Decision Tree) ให้ตรงโจทย์ 3–4"):
        ok = quick_train_decision_tree(df)
        if ok:
            st.experimental_rerun()

# ---------- แสดงผลการเทรน (ข้อ 4) ----------
with st.expander("📊 Training Results (หลาย features/parameters)"):
    if METRICS_CSV.exists():
        try:
            mdf = pd.read_csv(METRICS_CSV)
            if not mdf.empty:
                st.dataframe(mdf, use_container_width=True)
                top = mdf.sort_values("accuracy", ascending=False).iloc[0]
                st.success(
                    f"โมเดลที่เลือก: feature_set = {top['feature_set']}, "
                    f"params = {top['params']}, accuracy = {top['accuracy']}"
                )
            else:
                st.info("metrics.csv ว่าง — กด Quick Train หรือรัน train.py")
        except Exception as e:
            st.error(f"อ่าน metrics.csv ไม่สำเร็จ: {e}")
    else:
        st.info("ยังไม่พบ metrics.csv — กด Quick Train หรือรัน train.py")

# ---------- โหลดโมเดล (ถ้ามี) ----------
model, le = None, None
if MODEL_PATH.exists() and ENC_PATH.exists():
    try:
        model = joblib.load(MODEL_PATH)
        le    = joblib.load(ENC_PATH)
    except Exception as e:
        st.error(f"โหลดโมเดล/encoder ไม่สำเร็จ: {e}")
        st.code(traceback.format_exc())

# ---------- UI ส่วนเลือก/กรอก + ทำนาย ----------
if df is not None and model is not None and le is not None:
    # ทำความสะอาดสำหรับ UI
    for col in ["magnitude","depth","cdi","mmi","sig"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col].fillna(df[col].median(), inplace=True)

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

    # เผยแพร่ประกาศ
    if "last_pred" in st.session_state:
        st.divider()
        st.subheader("📢 เผยแพร่ประกาศ (ประชาชนจะเห็นด้านล่าง)")
        pred = st.session_state["last_pred"]
        region = st.text_input("พื้นที่/ภูมิภาค", value=pred["region"])
        msg_default = f"ตรวจพบเหตุสั่นสะเทือนระดับ {pred['risk'].upper()} — โปรดปฏิบัติตามคำแนะนำความปลอดภัย"
        message = st.text_area("ข้อความประกาศ", value=msg_default, height=80)
        tips = ["อยู่ใต้โต๊ะ/โครงสร้างแข็งแรง", "หลีกเลี่ยงลิฟต์/กระจก", "ปิดแก๊ส/ไฟฟ้า"]

        if st.button("✅ เผยแพร่ประกาศ", type="primary", use_container_width=True):
            STORE_DIR.mkdir(parents=True, exist_ok=True)
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

# พื้นที่ประชาชนเห็น
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

# Credits (ตอบข้อ 2)
with st.expander("ℹ️ Dataset Source / ข้อมูลอ้างอิง"):
    st.write("Dataset: Kaggle – Earthquake Alert Prediction (Ahmed Uzaki) หรือข้อมูลจริงจากหน่วยงานที่เกี่ยวข้อง (โปรดยืนยันแหล่งที่มาในรายงาน)")
