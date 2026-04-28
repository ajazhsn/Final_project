# app.py — Plant Disease Detection Streamlit Frontend
# DA5402 — Final Project
import io
import json
import zipfile
import requests
import streamlit as st
from PIL import Image
from datetime import datetime

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
API_BASE = "http://127.0.0.1:8000"

TOOL_LINKS = {
    "MLflow": "http://localhost:5000",
    "Grafana": "http://localhost:3000",
    "Prometheus": "http://localhost:9090",
    "Airflow": "http://localhost:8080",
    "FastAPI Docs": "http://localhost:8000/docs",
    "DagsHub": "https://dagshub.com/ajazhsn/Final_project",
}

st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# CSS Styling
# ─────────────────────────────────────────
st.markdown("""
<style>
    /* Base */
    .stApp {
        background: #0f1117;
        color: #e0e0e0;
    }

    [data-testid="stSidebar"] {
        background: #161b22;
        border-right: 1px solid #30363d;
    }

    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #1a1f2e 0%, #1e2d40 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        border: 1px solid #30363d;
        border-left: 4px solid #4f9cf9;
    }
    .main-header h1 {
        color: #e6edf3;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #8b949e;
        font-size: 1rem;
        margin: 0.4rem 0 0 0;
    }

    /* Cards */
    .result-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 0.8rem 0;
    }
    .result-card h3 {
        color: #e6edf3;
        font-size: 1.2rem;
        margin: 0 0 1rem 0;
        padding-bottom: 0.7rem;
        border-bottom: 1px solid #30363d;
    }
    .result-card p {
        color: #8b949e;
        margin: 0.4rem 0;
        font-size: 0.95rem;
    }
    .result-card p b {
        color: #c9d1d9;
    }

    /* Metric cards */
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        color: #4f9cf9;
        font-size: 1.6rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .metric-label {
        color: #8b949e;
        font-size: 0.8rem;
        margin-top: 0.2rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Severity */
    .severity-high {
        background: #1f1117;
        border-left: 3px solid #f85149;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        color: #ffa198;
        margin: 0.5rem 0;
    }
    .severity-medium {
        background: #1f1a0f;
        border-left: 3px solid #d29922;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        color: #e3b341;
        margin: 0.5rem 0;
    }
    .severity-none {
        background: #0f1f14;
        border-left: 3px solid #3fb950;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        color: #7ee787;
        margin: 0.5rem 0;
    }

    /* Status badges */
    .badge-healthy {
        background: #0f1f14;
        color: #3fb950;
        padding: 0.3rem 0.9rem;
        border-radius: 20px;
        font-size: 0.82rem;
        border: 1px solid #3fb950;
        font-weight: 600;
        display: inline-block;
    }
    .badge-offline {
        background: #1f0f0f;
        color: #f85149;
        padding: 0.3rem 0.9rem;
        border-radius: 20px;
        font-size: 0.82rem;
        border: 1px solid #f85149;
        font-weight: 600;
        display: inline-block;
    }

    /* Section headers */
    .section-header {
        color: #8b949e;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        border-bottom: 1px solid #21262d;
        padding-bottom: 0.5rem;
        margin: 1.2rem 0 0.8rem 0;
    }

    /* Upload area */
    [data-testid="stFileUploader"] {
        background: #161b22;
        border: 2px dashed #30363d;
        border-radius: 10px;
        padding: 1rem;
    }

    /* Tool cards */
    .tool-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 1.3rem;
        margin-bottom: 1rem;
        transition: border-color 0.2s;
    }
    .tool-card:hover {
        border-color: #4f9cf9;
    }
    .tool-card h3 {
        color: #e6edf3;
        margin: 0 0 0.4rem 0;
        font-size: 1rem;
        font-weight: 600;
    }
    .tool-card p {
        color: #8b949e;
        font-size: 0.85rem;
        margin: 0 0 0.8rem 0;
    }
    .tool-btn {
        display: inline-block;
        background: #21262d;
        color: #4f9cf9 !important;
        text-decoration: none !important;
        padding: 0.45rem 1rem;
        border-radius: 6px;
        border: 1px solid #30363d;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.3px;
    }
    .tool-btn:hover {
        background: #1f6feb20;
        border-color: #4f9cf9;
    }

    /* Sidebar text */
    .sidebar-title {
        color: #e6edf3;
        font-size: 1.1rem;
        font-weight: 700;
        margin: 0.5rem 0 0.2rem 0;
    }
    .sidebar-sub {
        color: #8b949e;
        font-size: 0.78rem;
    }

    /* Hide defaults */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Scrollbar */
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: #0d1117; }
    ::-webkit-scrollbar-thumb {
        background: #30363d;
        border-radius: 3px;
    }

    /* Progress bars */
    .stProgress > div > div {
        background: #4f9cf9;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        color: #8b949e;
    }
    .stTabs [aria-selected="true"] {
        color: #4f9cf9 !important;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────
def check_api_health():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def predict_single(image_bytes, filename):
    try:
        r = requests.post(
            f"{API_BASE}/predict",
            files={"file": (filename, image_bytes, "image/jpeg")},
            timeout=30
        )
        return r.json() if r.status_code == 200 else None
    except Exception as e:
        return {"error": str(e)}


def predict_batch(files):
    try:
        r = requests.post(
            f"{API_BASE}/predict/batch",
            files=[("files", (f[0], f[1], "image/jpeg"))
                   for f in files],
            timeout=60
        )
        return r.json() if r.status_code == 200 else None
    except Exception as e:
        return {"error": str(e)}


def get_disease_info(class_name):
    info = {
        "Potato___Early_blight": {
            "severity": "Medium", "emoji": "🟡",
            "treatment": "Apply fungicide containing chlorothalonil. Remove infected leaves. Ensure proper spacing.",
            "description": "Caused by Alternaria solani fungus. Brown spots with yellow rings."
        },
        "Potato___Late_blight": {
            "severity": "High", "emoji": "🔴",
            "treatment": "Apply copper-based fungicide immediately. Destroy infected plants. Avoid overhead irrigation.",
            "description": "Caused by Phytophthora infestans. Dark water-soaked lesions."
        },
        "Potato___healthy": {
            "severity": "None", "emoji": "🟢",
            "treatment": "Plant is healthy! Continue regular care and monitoring.",
            "description": "No disease detected. Plant appears healthy."
        },
        "Tomato_Early_blight": {
            "severity": "Medium", "emoji": "🟡",
            "treatment": "Use fungicide spray. Remove lower infected leaves. Mulch around base of plant.",
            "description": "Caused by Alternaria solani. Dark spots with concentric rings."
        },
        "Tomato_Late_blight": {
            "severity": "High", "emoji": "🔴",
            "treatment": "Apply fungicide immediately. Remove and destroy infected plants. Improve air circulation.",
            "description": "Caused by Phytophthora infestans. Brown-black lesions on leaves and stems."
        },
        "Tomato_healthy": {
            "severity": "None", "emoji": "🟢",
            "treatment": "Plant is healthy! Continue regular care and monitoring.",
            "description": "No disease detected. Plant appears healthy."
        }
    }
    return info.get(class_name, {
        "severity": "Unknown", "emoji": "⚪",
        "treatment": "Consult an agricultural expert.",
        "description": "Unknown plant condition."
    })


def severity_class(severity):
    if severity == "High":
        return "severity-high"
    elif severity == "Medium":
        return "severity-medium"
    return "severity-none"


# ─────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1.2rem 0 0.5rem 0;'>
        <div style='font-size: 2rem;'>🌿</div>
        <p class="sidebar-title">Plant Doctor</p>
        <p class="sidebar-sub">AI Disease Detection System</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # API Health
    health = check_api_health()
    if health:
        st.markdown(
            '<span class="badge-healthy">✅ API Connected</span>',
            unsafe_allow_html=True)
        st.caption(f"Model: PlantDiseaseDetector@champion")
        st.caption(f"Classes: {health['num_classes']}")
        st.caption(f"Container: {health['container_id'][:12]}")
    else:
        st.markdown(
            '<span class="badge-offline">❌ API Offline</span>',
            unsafe_allow_html=True)
        st.caption("Start the FastAPI backend first")

    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🔍 Single Detection",
         "📦 Bulk Detection",
         "📊 Pipeline Console",
         "🔧 MLOps Tools",
         "ℹ️ About"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style='padding: 0.8rem 0; border-top: 1px solid #21262d;'>
        <p style='color: #8b949e; font-size: 0.75rem; margin: 0;'>
        DA5402 · MLOps Final Project<br>
        <span style='color: #4f9cf9; font-weight: 600;'>
        Test F1: 0.9083</span>
        </p>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────
# Page: Single Detection
# ─────────────────────────────────────────
if page == "🔍 Single Detection":
    st.markdown("""
    <div class="main-header">
        <h1>🌿 Plant Disease Detector</h1>
        <p>Upload a leaf image for instant AI-powered disease diagnosis</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<p class="section-header">📸 Upload Image</p>',
                    unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Choose a leaf image",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear photo of a plant leaf",
            label_visibility="collapsed"
        )

        if uploaded:
            image = Image.open(uploaded)
            st.image(image, caption="Uploaded Leaf Image",
                     use_container_width=True)

            # Image info
            st.markdown(f"""
            <div class="metric-card">
                <div style='display:flex; justify-content:space-around;'>
                    <div>
                        <div class="metric-value">{image.size[0]}px</div>
                        <div class="metric-label">Width</div>
                    </div>
                    <div>
                        <div class="metric-value">{image.size[1]}px</div>
                        <div class="metric-label">Height</div>
                    </div>
                    <div>
                        <div class="metric-value">{uploaded.size//1024}KB</div>
                        <div class="metric-label">Size</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown('<p class="section-header">🔬 Analysis Result</p>',
                    unsafe_allow_html=True)

        if uploaded:
            with st.spinner("🔬 Analyzing leaf..."):
                img_bytes = uploaded.getvalue()
                result = predict_single(img_bytes, uploaded.name)

            if result and "error" not in result:
                disease_info = get_disease_info(
                    result["predicted_class"])
                sev = disease_info["severity"]

                # Main result card
                st.markdown(f"""
                <div class="result-card">
                    <h3>{disease_info['emoji']} {result['predicted_class'].replace('___', ' — ').replace('_', ' ')}</h3>
                    <p>🎯 <b>Confidence:</b> {result['confidence']*100:.1f}%</p>
                    <p>⚠️ <b>Severity:</b> {sev}</p>
                    <p>📋 <b>Description:</b> {disease_info['description']}</p>
                </div>
                """, unsafe_allow_html=True)

                # Treatment
                st.markdown(f"""
                <div class="{severity_class(sev)}">
                    <b>💊 Recommended Treatment:</b><br>
                    {disease_info['treatment']}
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Performance metrics
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{result['confidence']*100:.1f}%</div>
                        <div class="metric-label">Confidence</div>
                    </div>""", unsafe_allow_html=True)
                with m2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{result['inference_time_ms']:.0f}ms</div>
                        <div class="metric-label">Inference Time</div>
                    </div>""", unsafe_allow_html=True)
                with m3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{result['container_id'][:8]}</div>
                        <div class="metric-label">Container</div>
                    </div>""", unsafe_allow_html=True)

                # Probabilities
                # Probabilities
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(
                    '<p class="section-header">📊 All Class Probabilities</p>',
                    unsafe_allow_html=True)
                probs = result["all_probabilities"]
                for cls, prob in sorted(probs.items(),
                                        key=lambda x: x[1],
                                        reverse=True):
                    label = cls.replace(
                        "___", " — ").replace("_", " ")
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.progress(prob)
                    with col_b:
                        st.markdown(
                            f"<p style='color:#c9d1d9; margin:0; padding-top:0.3rem'>"
                            f"{label}: {prob*100:.1f}%</p>",
                            unsafe_allow_html=True)

            elif result and "error" in result:
                st.error(f"❌ Error: {result['error']}")
        else:
            st.markdown("""
            <div style='text-align:center; padding: 3rem 1rem;
                        border: 2px dashed #2d5a2d; border-radius: 12px;
                        color: #6a9a6a;'>
                <div style='font-size: 3rem;'>🍃</div>
                <h3 style='color: #90ee90;'>Upload a leaf image</h3>
                <p>Supported plants:</p>
                <p>🥔 Potato (Early Blight, Late Blight, Healthy)</p>
                <p>🍅 Tomato (Early Blight, Late Blight, Healthy)</p>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────
# Page: Bulk Detection
# ─────────────────────────────────────────
elif page == "📦 Bulk Detection":
    st.markdown("""
    <div class="main-header">
        <h1>📦 Bulk Detection</h1>
        <p>Analyze multiple leaf images simultaneously</p>
    </div>
    """, unsafe_allow_html=True)

    upload_type = st.radio(
        "Upload Type",
        ["🖼️ Multiple Images", "🗜️ ZIP File"],
        horizontal=True
    )

    files_to_process = []

    if upload_type == "🖼️ Multiple Images":
        uploaded_files = st.file_uploader(
            "Upload images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        if uploaded_files:
            files_to_process = [
                (f.name, f.getvalue()) for f in uploaded_files]
    else:
        zip_file = st.file_uploader(
            "Upload ZIP", type=["zip"],
            label_visibility="collapsed")
        if zip_file:
            with zipfile.ZipFile(
                    io.BytesIO(zip_file.getvalue())) as zf:
                for name in zf.namelist():
                    if name.lower().endswith(
                            ('.jpg', '.jpeg', '.png')):
                        files_to_process.append(
                            (name, zf.read(name)))
            st.info(
                f"📁 Found {len(files_to_process)} images in ZIP")

    if files_to_process:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(files_to_process)}</div>
            <div class="metric-label">Images ready to process</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🚀 Run Batch Analysis", type="primary",
                     use_container_width=True):
            with st.spinner(
                    f"Processing {len(files_to_process)} images..."):
                result = predict_batch(files_to_process)

            if result and "error" not in result:
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{result['total_images']}</div>
                        <div class="metric-label">Total</div>
                    </div>""", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value" style="color:#90ee90">{result['successful']}</div>
                        <div class="metric-label">Successful</div>
                    </div>""", unsafe_allow_html=True)
                with c3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value" style="color:#ff9090">{result['failed']}</div>
                        <div class="metric-label">Failed</div>
                    </div>""", unsafe_allow_html=True)
                with c4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{result['total_time_ms']:.0f}ms</div>
                        <div class="metric-label">Total Time</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(
                    '<p class="section-header">📋 Results</p>',
                    unsafe_allow_html=True)

                for r in result["results"]:
                    if r["status"] == "success":
                        info = get_disease_info(
                            r["predicted_class"])
                        with st.expander(
                                f"{info['emoji']} {r['filename']} — "
                                f"{r['predicted_class'].replace('___', ' ').replace('_', ' ')} "
                                f"({r['confidence']*100:.1f}%)"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(
                                    f"**Disease:** {r['predicted_class']}")
                                st.write(
                                    f"**Confidence:** {r['confidence']*100:.1f}%")
                                st.write(
                                    f"**Severity:** {info['severity']}")
                            with col2:
                                st.write(
                                    f"**Treatment:** {info['treatment']}")
                    else:
                        st.error(
                            f"❌ {r['filename']}: "
                            f"{r.get('error', 'Failed')}")


# ─────────────────────────────────────────
# Page: Pipeline Console
# ─────────────────────────────────────────
elif page == "📊 Pipeline Console":
    st.markdown("""
    <div class="main-header">
        <h1>📊 Pipeline Console</h1>
        <p>Monitor model performance and pipeline health</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([
        "🏆 Model Performance",
        "📡 Live API Metrics",
        "🗄️ Database Stats"
    ])

    with tab1:
        try:
            with open("metrics/final_metrics.json") as f:
                final = json.load(f)

            st.markdown(f"""
            <div class="result-card">
                <h3>🏆 Best Model: {final['best_model']}</h3>
                <p>✅ Test Macro F1: <b>{final['test_macro_f1']:.4f}</b></p>
                <p>✅ Test Accuracy: <b>{final['test_accuracy']:.4f}</b></p>
                <p>📅 Evaluated: {final['evaluated_at'][:10]}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(
                '<p class="section-header">📊 Model Comparison</p>',
                unsafe_allow_html=True)

            for m in final.get("all_models", []):
                is_best = m['model'] == final['best_model']
                color = "#3fb950" if is_best else "#4f9cf9"
                st.markdown(f"""
                <div style="display:flex; align-items:center;
                            margin:0.6rem 0; gap:1rem;">
                    <div style="width:130px; color:#c9d1d9;
                                font-weight:{'700' if is_best else '400'};
                                font-size:0.9rem;">
                        {m['model']} {'🏆' if is_best else ''}
                    </div>
                    <div style="flex:1; background:#21262d;
                                border-radius:4px; height:26px;
                                overflow:hidden;">
                        <div style="background:{color};
                                    width:{m['f1']*100:.1f}%;
                                    height:100%; border-radius:4px;
                                    display:flex; align-items:center;
                                    padding-left:8px;">
                            <span style="color:white; font-size:0.82rem;
                                         font-weight:600;">
                                F1: {m['f1']:.4f}
                            </span>
                        </div>
                    </div>
                    <div style="width:90px; color:#8b949e;
                                font-size:0.85rem; text-align:right;">
                        Acc: {m['accuracy']:.4f}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                '<p class="section-header">📈 Per-Class F1 Scores</p>',
                unsafe_allow_html=True)

            report = final.get("classification_report", {})
            for cls, scores in report.items():
                if isinstance(scores, dict) and \
                        "f1-score" in scores and \
                        cls not in ["macro avg",
                                    "weighted avg",
                                    "accuracy"]:
                    label = cls.replace(
                        "___", " — ").replace("_", " ")
                    f1 = scores["f1-score"]
                    color = "#3fb950" if f1 > 0.9 else \
                            "#d29922" if f1 > 0.8 else "#f85149"
                    st.markdown(f"""
                    <div style="margin:0.5rem 0;">
                        <div style="display:flex;
                                    justify-content:space-between;
                                    margin-bottom:0.25rem;">
                            <span style="color:#c9d1d9;
                                         font-size:0.85rem;">
                                {label}</span>
                            <span style="color:{color};
                                         font-weight:600;
                                         font-size:0.85rem;">
                                F1={f1:.3f}</span>
                        </div>
                        <div style="background:#21262d;
                                    border-radius:4px; height:8px;
                                    overflow:hidden;">
                            <div style="background:{color};
                                        width:{f1*100:.1f}%;
                                        height:100%;
                                        border-radius:4px;">
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        except FileNotFoundError:
            st.warning("Run `python3 src/evaluate.py` first")

    with tab2:
        if st.button("🔄 Refresh Metrics"):
            try:
                r = requests.get(
                    f"{API_BASE}/metrics", timeout=5)
                lines = [
                    l for l in r.text.split("\n")
                    if not l.startswith("#") and
                    "plant_" in l and l.strip()
                ]
                st.code("\n".join(lines[:25]),
                        language="text")
            except Exception:
                st.error("Could not fetch metrics from API")

        health = check_api_health()
        if health:
            st.markdown(
                '<p class="section-header">🏥 API Health</p>',
                unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value"
                         style="color:#3fb950">
                    {health['status'].upper()}</div>
                    <div class="metric-label">Status</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">
                    {health['num_classes']}</div>
                    <div class="metric-label">Classes</div>
                </div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">
                    {health['container_id'][:8]}</div>
                    <div class="metric-label">Container</div>
                </div>""", unsafe_allow_html=True)

    with tab3:
        try:
            import sqlite3
            conn = sqlite3.connect("data/plantdisease.db")
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM predictions "
                "WHERE status='success'")
            total = cursor.fetchone()[0]
            cursor.execute(
                "SELECT predicted_class, COUNT(*) "
                "FROM predictions WHERE status='success' "
                "GROUP BY predicted_class "
                "ORDER BY COUNT(*) DESC")
            distribution = cursor.fetchall()
            cursor.execute(
                "SELECT AVG(confidence) FROM predictions "
                "WHERE status='success'")
            avg_conf = cursor.fetchone()[0] or 0
            conn.close()

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{total}</div>
                    <div class="metric-label">
                    Total Predictions</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">
                    {avg_conf*100:.1f}%</div>
                    <div class="metric-label">
                    Avg Confidence</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                '<p class="section-header">'
                '📊 Prediction Distribution</p>',
                unsafe_allow_html=True)

            for cls, count in distribution:
                label = cls.replace(
                    "___", " — ").replace("_", " ")
                pct = count / max(total, 1)
                st.markdown(f"""
                <div style="margin:0.5rem 0;">
                    <div style="display:flex;
                                justify-content:space-between;
                                margin-bottom:0.25rem;">
                        <span style="color:#c9d1d9;
                                     font-size:0.85rem;">
                            {label}</span>
                        <span style="color:#4f9cf9;
                                     font-weight:600;
                                     font-size:0.85rem;">
                            {count} predictions</span>
                    </div>
                    <div style="background:#21262d;
                                border-radius:4px; height:8px;
                                overflow:hidden;">
                        <div style="background:#4f9cf9;
                                    width:{pct*100:.1f}%;
                                    height:100%;
                                    border-radius:4px;">
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.warning(f"Database not available: {e}")

# ─────────────────────────────────────────
# Page: MLOps Tools
# ─────────────────────────────────────────
elif page == "🔧 MLOps Tools":
    st.markdown("""
    <div class="main-header">
        <h1>🔧 MLOps Tools</h1>
        <p>Access all monitoring and management interfaces</p>
    </div>
    """, unsafe_allow_html=True)

    # Tool cards
    tools = [
        {
            "name": "MLflow",
            "icon": "🔬",
            "url": "http://localhost:5000",
            "desc": "Experiment tracking, model registry, run comparison",
            "color": "#1a3a4a"
        },
        {
            "name": "Grafana",
            "icon": "📊",
            "url": "http://localhost:3000",
            "desc": "Real-time monitoring dashboard, metrics visualization",
            "color": "#2a1a3a"
        },
        {
            "name": "Prometheus",
            "icon": "📡",
            "url": "http://localhost:9090",
            "desc": "Metrics collection, PromQL queries, alert rules",
            "color": "#3a2a1a"
        },
        {
            "name": "Airflow",
            "icon": "🔄",
            "url": "http://localhost:8080",
            "desc": "Pipeline orchestration, DAG management, task logs",
            "color": "#1a3a2a"
        },
        {
            "name": "FastAPI Docs",
            "icon": "⚡",
            "url": "http://localhost:8000/docs",
            "desc": "Interactive API documentation and testing",
            "color": "#1a2a3a"
        },
        {
            "name": "AlertManager",
            "icon": "🚨",
            "url": "http://localhost:9093",
            "desc": "Alert routing, silences, email notifications",
            "color": "#3a1a1a"
        },
        {
            "name": "DagsHub",
            "icon": "🐱",
            "url": "https://dagshub.com/ajazhsn/Final_project",
            "desc": "Data versioning, DVC remote, experiment history",
            "color": "#1a1a3a"
        },
        {
            "name": "GitHub",
            "icon": "💻",
            "url": "https://github.com/ajazhsn/Final_project",
            "desc": "Source code, commits, branches, pull requests",
            "color": "#2a2a2a"
        },
    ]

    col1, col2 = st.columns(2)
    for i, tool in enumerate(tools):
        with col1 if i % 2 == 0 else col2:
            st.markdown(f"""
            <div class="tool-card">
                <h3>{tool['icon']} {tool['name']}</h3>
                <p>{tool['desc']}</p>
                <a href="{tool['url']}" target="_blank"
                   class="tool-btn">
                    Open {tool['name']} ↗
                </a>
            </div>
            """, unsafe_allow_html=True)

    # Quick status check
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<p class="section-header">🏥 Service Status</p>',
        unsafe_allow_html=True)

    services = [
        ("FastAPI Backend", "http://localhost:8000/health"),
        ("MLflow Server", "http://localhost:5000/health"),
        ("Prometheus", "http://localhost:9090/-/healthy"),
        ("Grafana", "http://localhost:3000/api/health"),
        ("Airflow", "http://localhost:8080/api/v1/health"),
        ("AlertManager", "http://localhost:9093/-/healthy"),
    ]

    cols = st.columns(3)
    for i, (name, url) in enumerate(services):
        try:
            r = requests.get(url, timeout=2)
            status = "✅ Online" if r.status_code == 200 else "⚠️ Issue"
            color = "#90ee90" if r.status_code == 200 else "#ffaa00"
        except Exception:
            status = "❌ Offline"
            color = "#ff9090"

        with cols[i % 3]:
            st.markdown(f"""
            <div class="metric-card" style="margin-bottom: 0.5rem;">
                <div style="color: {color}; font-weight: 600;">
                {status}</div>
                <div class="metric-label">{name}</div>
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# Page: About
# ─────────────────────────────────────────
elif page == "ℹ️ About":
    st.markdown("""
    <div class="main-header">
        <h1>ℹ️ About This Project</h1>
        <p>DA5402 — Graduate Machine Learning Operations</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="result-card">
            <h3>🌿 Plant Disease Detection System</h3>
            <p>An end-to-end MLOps application that identifies
            plant diseases from leaf images using deep learning.</p>
            <br>
            <p>📊 <b>Model:</b> Custom CNN (3-block)</p>
            <p>🎯 <b>Test F1:</b> 0.9083 (>0.90 threshold)</p>
            <p>⚡ <b>Inference:</b> ~300ms per image</p>
            <p>🔢 <b>Classes:</b> 6 (Potato + Tomato)</p>
            <p>🖼️ <b>Dataset:</b> PlantVillage (6,651 images)</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="result-card">
            <h3>🏗️ Architecture</h3>
            <p>🌐 <b>Frontend:</b> Streamlit (port 8501)</p>
            <p>⚡ <b>Backend:</b> FastAPI × 3 (8000-8002)</p>
            <p>🔬 <b>Tracking:</b> MLflow (port 5000)</p>
            <p>📡 <b>Metrics:</b> Prometheus (port 9090)</p>
            <p>📊 <b>Dashboard:</b> Grafana (port 3000)</p>
            <p>🔄 <b>Pipeline:</b> Airflow (port 8080)</p>
            <p>📦 <b>Data:</b> DVC + DagsHub</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="result-card">
            <h3>📊 Model Comparison</h3>
            <p>🔴 <b>SVM + HOG:</b> F1 = 0.6132</p>
            <p>🟡 <b>MLP:</b> F1 = 0.5843</p>
            <p>🟢 <b>Custom CNN:</b> F1 = 0.9083 ✅</p>
            <br>
            <p>The Custom CNN achieves >0.90 F1 score,
            placing it in the maximum performance bracket.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="result-card">
            <h3>🧪 Testing</h3>
            <p>✅ 19/19 unit tests passing</p>
            <p>✅ Health endpoint verified</p>
            <p>✅ Prediction accuracy verified</p>
            <p>✅ Probability sum verified</p>
            <p>✅ Error handling verified</p>
            <p>✅ Metrics endpoint verified</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="result-card">
            <h3>👨‍💻 Project Info</h3>
            <p>👤 <b>Student:</b> Ajaz Hussain</p>
            <p>🎓 <b>ID:</b> DA25M006</p>
            <p>📚 <b>Course:</b> DA5402 MLOps</p>
            <p>🔗 <b>GitHub:</b> ajazhsn/Final_project</p>
            <p>📅 <b>Year:</b> 2026</p>
        </div>
        """, unsafe_allow_html=True)
