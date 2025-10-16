# AI_project_master/app.py
import os, json, time, base64
from pathlib import Path
from datetime import datetime

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

# (PIN แบบง่ายสำหรับงานส่ง — เปลี่ยนได้ด้วย ENV)
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
    st.caption("แดชบอร์ดสำหรับเจ้าหน้าที่: ดูความแม่นยำ/พารามิเตอร์/ฟีเจอร์สำคัญ/เมทริกซ์ความสับสน + Export รายงาน")

# ---------------- Common: Check files ----------------
missing = [p for p in [DATA_PATH, MODEL_PATH, ENC_PATH] if not p.exists()]
if mode == "Public Alert":
    if missing:
        st.error("ไม่พบไฟล์ต่อไปนี้:\n" + "\n".join(f"- {p}" for p in missing))
        st.stop()
else:
    pass  # ในหน้า Admin แสดงเท่าที่มี

# ---------------- Load (if exists) ----------------
df = pd.read_csv(DATA_PATH) if DATA_PATH.exists() else None
model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
le    = joblib.load(ENC_PATH)   if ENC_PATH.exists()   else None

# ---------- Helpers ----------
def file_mtime(path: Path) -> str:
    if not path.exists(): return "-"
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime))

def read_file_bytes(path: Path) -> bytes:
    return path.read_bytes() if path.exists() else b""

def img_to_base64(path: Path) -> str:
    if not path.exists(): return ""
    return base64.b64encode(path.read_bytes()).decode("utf-8")

def build_admin_report_html(metrics_df: pd.DataFrame | None,
                            acc: float | None,
                            best_info: dict,
                            feature_names: list[str],
                            feature_importances: list[float] | None,
                            cm_base64: str) -> str:
    # Styles
    style = """
    <style>
      body { font-family: Arial, sans-serif; margin: 24px; }
      h1, h2 { margin-bottom: 8px; }
      .pill { display:inline-block; padding:4px 10px; border-radius:999px; background:#eef; margin-right:8px; }
      table { border-collapse: collapse; width: 100%; margin: 8px 0 16px 0;}
      th, td { border: 1px solid #ddd; padding: 8px; text-align:left;}
      th { background:#f7f7f7; }
      .small { color:#666; font-size: 12px; }
      .section { margin: 18px 0; }
      .muted { color:#555; }
      .tag { background:#f3f3f3; padding:3px 8px; border-radius:6px; margin-right:6px; }
    </style>
    """
    updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    acc_str = f"{acc*100:.2f}%" if acc is not None else "-"

    # Feature importance table
    fi_html = ""
    if feature_importances and feature_names:
        rows = "".join(
            f"<tr><td>{n}</td><td>{round(v,4)}</td></tr>"
            for n, v in sorted(zip(feature_names, feature_importances), key=lambda x: x[1], reverse=True)
        )
        fi_html = f"""
        <div class="section">
          <h2>Feature Importances</h2>
          <table>
            <thead><tr><th>Feature</th><th>Importance</th></tr></thead>
            <tbody>{rows}</tbody>
          </table>
        </div>
        """

    # Best params/feature-set
    bp = best_info
    params_html = ""
    if bp.get("feature_set") or bp.get("features") or bp.get("params"):
        params_html = f"""
        <div class="section">
          <h2>Best Configuration</h2>
          <div><span class="tag">feature_set</span> {bp.get('feature_set','-')}</div>
          <div><span class="tag">features</span> {bp.get('features','-')}</div>
          <div><span class="tag">params</span> {bp.get('params','-')}</div>
        </div>
        """

    # Confusion matrix image
    cm_html = ""
    if cm_base64:
        cm_html = f"""
        <div class="section">
          <h2>Confusion Matrix</h2>
          <img src="data:image/png;base64,{cm_base64}" style="max-width:100%;border:1px solid #ddd;border-radius:8px"/>
        </div>
        """

    # metrics table (top 10)
    metrics_table = ""
    if metrics_df is not None and not metrics_df.empty:
        top10 = metrics_df.sort_values("accuracy", ascending=False).head(10)
        rows = "".join(
            f"<tr><td>{r.feature_set}</td><td>{r.features}</td><td>{r.params}</td><td>{r.accuracy:.4f}</td></tr>"
            for r in top10.itertuples(index=False)
        )
        metrics_table = f"""
        <div class="section">
          <h2>Top Experiments (by Accuracy)</h2>
          <table>
            <thead><tr><th>feature_set</th><th>features</th><th>params</th><th>accuracy</th></tr></thead>
            <tbody>{rows}</tbody>
          </table>
        </div>
        """

    html = f"""
    <!doctype html>
    <html><head><meta charset="utf-8"/>{style}<title>Admin Report - Earthquake AI</title></head>
    <body>
      <h1>Earthquake AI — Admin Report</h1>
      <div class="small">Generated at: {updated}</div>
      <div class="section">
        <span class="pill">Model: Decision Tree</span>
        <span class="pill">Test Accuracy: {acc_str}</span>
      </div>
      {params_html}
      {fi_html}
      {cm_html}
      {metrics_table}
      <div class="section muted small">
        Notes: Accuracy/FI อ้างอิงจากไฟล์ metrics.csv และโมเดลล่าสุดที่บันทึกไว้ในโฟลเดอร์ models/
      </div>
    </body></html>
    """
    return html

# ========== MODE: ADMIN DASHBOARD ==========
if mode == "Admin Dashboard":
    if not admin_ok:
        st.warning("โปรดกรอก Admin PIN ทางแถบซ้ายเพื่อเข้าถึงแดชบอร์ด")
        st.stop()

    colA, colB, colC = st.columns([1,1,1])

    # 1) Accuracy & Best Params from metrics.csv
    with colA:
        st.subheader("📊 Model Accuracy")
        acc = None
        top_row = {}
        mdf = None
        if METRICS_CSV.exists():
            mdf = pd.read_csv(METRICS_CSV)
            if not mdf.empty and "accuracy" in mdf.columns:
                top_row = mdf.sort_values("accuracy", ascending=False).iloc[0].to_dict()
                acc = float(top_row["accuracy"])
                st.metric("Accuracy (Test)", f"{acc*100:.2f}%")
            else:
                st.info("metrics.csv ว่างหรือไม่มีคอลัมน์ accuracy")
        else:
            st.info("ยังไม่พบ metrics.csv (สร้างจากโน้ตบุ๊กหรือ train.py)")

    with colB:
        st.subheader("⚙️ Best Parameters")
        if top_row:
            st.code(
                f"feature_set: {top_row.get('feature_set','-')}\n"
                f"features   : {top_row.get('features','-')}\n"
                f"params     : {top_row.get('params','-')}",
                language="text"
            )
        else:
            st.info("ยังไม่มีข้อมูล best config")

    with colC:
        st.subheader("🕒 Model Files")
        rows = []
        for p in [MODEL_PATH, ENC_PATH, METRICS_CSV]:
            rows.append({
                "file": str(p.name),
                "status": "OK" if p.exists() else "missing",
                "updated": file_mtime(p)
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.divider()

    # 2) Feature Importances (Decision Tree)
    st.subheader("🌳 Feature Importances (Decision Tree)")
    feat_cols = [c for c in ["magnitude","depth","cdi","mmi","sig"] if df is not None and c in df.columns]
    if model is not None and hasattr(model, "feature_importances_") and feat_cols:
        try:
            imps = pd.Series(model.feature_importances_, index=feat_cols).sort_values(ascending=False)
            st.bar_chart(imps)  # แสดงเฉพาะหน้า admin
            st.caption("หมายเหตุ: ค่า importance สูง = โมเดลให้ความสำคัญกับฟีเจอร์นั้นมากในการตัดสินใจ")
        except Exception as e:
            st.error(f"ไม่สามารถอ่าน feature_importances_: {e}")
    else:
        st.info("ยังไม่พบโมเดลหรือตัวชี้วัดความสำคัญของฟีเจอร์ (ต้องใช้ Decision Tree)")

    # 3) Confusion Matrix (from notebook if exists)
    st.subheader("🧩 Confusion Matrix (จากการประเมินในโน้ตบุ๊ก)")
    if FIG_CM.exists():
        st.image(str(FIG_CM), caption="Confusion Matrix (ไฟล์จาก notebooks)")
    else:
        st.info("ยังไม่พบไฟล์รูป Confusion Matrix (metrics/figs/confusion_matrix.png)")

    st.divider()

    # 4) Export Report & Data
    st.subheader("📤 Export รายงานสำหรับเจ้าหน้าที่")
    # 4.1 ดาวน์โหลด metrics.csv ต้นฉบับ
    if METRICS_CSV.exists():
        st.download_button(
            label="⬇️ ดาวน์โหลด metrics.csv",
            data=read_file_bytes(METRICS_CSV),
            file_name="metrics.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("ยังไม่มี metrics.csv สำหรับดาวน์โหลด")

    # 4.2 ดาวน์โหลด HTML Report (รวม Accuracy/Params/FI/CM/Top10)
    best_info = {
        "feature_set": top_row.get("feature_set") if top_row else None,
        "features": top_row.get("features") if top_row else None,
        "params": top_row.get("params") if top_row else None,
    }
    cm64 = img_to_base64(FIG_CM)
    fi_list = model.feature_importances_.tolist() if (model is not None and hasattr(model, "feature_importances_") and feat_cols) else None
    html_report = build_admin_report_html(
        metrics_df=mdf if (METRICS_CSV.exists() and mdf is not None) else None,
        acc=acc,
        best_info=best_info,
        feature_names=feat_cols,
        feature_importances=fi_list,
        cm_base64=cm64
    )
    st.download_button(
        label="📄 ดาวน์โหลด Admin Report (HTML)",
        data=html_report.encode("utf-8"),
        file_name=f"admin_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
        mime="text/html",
        use_container_width=True
    )

    st.caption("Tip: เปิดไฟล์ HTML นี้บนเบราว์เซอร์หรือแนบส่งให้คณะ/อาจารย์ได้ทันที")

    st.divider()
    st.subheader("📝 หมายเหตุสำหรับเจ้าหน้าที่")
    st.write(
        "- Accuracy เป็นผลจากชุดทดสอบ/การทดลอง (อ้างอิง metrics.csv)\n"
        "- แนะนำติดตาม Precision/Recall รายคลาสในโน้ตบุ๊กเพื่อดูจุดสับสน\n"
        "- ปรับปรุงโมเดล: อัปเดตรายการฟีเจอร์/พารามิเตอร์ใน notebooks/train.py แล้วสร้างไฟล์ใหม่\n"
    )

# ========== MODE: PUBLIC ALERT ==========
if mode == "Public Alert":
    # ทำความสะอาดตัวเลขสำหรับ UI
    for col in ["magnitude","depth","cdi","mmi","sig"]:
        if df is not None and col in df.columns:
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
