# app.py — Plant Disease Detection Streamlit Frontend
import io
import zipfile
import requests
import streamlit as st
from PIL import Image
from datetime import datetime

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
API_BASE = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# Styling
# ─────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2E7D32;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        background: #f0f7f0;
        border-left: 4px solid #2E7D32;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
        color: #1a1a1a;
    }
    .result-box h3 {
        color: #1a1a1a;
        margin-bottom: 0.5rem;
    }
    .result-box p {
        color: #1a1a1a;
        margin: 0.3rem 0;
    }
    .error-box {
        background: #fff3f3;
        border-left: 4px solid #c62828;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
        color: #1a1a1a;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
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
            files={"file": (filename, image_bytes,
                            "image/jpeg")},
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
            "severity": "Medium",
            "color": "🟡",
            "treatment": (
                "Apply fungicide containing chlorothalonil. "
                "Remove infected leaves. Ensure proper spacing."
            )
        },
        "Potato___Late_blight": {
            "severity": "High",
            "color": "🔴",
            "treatment": (
                "Apply copper-based fungicide immediately. "
                "Destroy infected plants. Avoid overhead irrigation."
            )
        },
        "Potato___healthy": {
            "severity": "None",
            "color": "🟢",
            "treatment": "Plant is healthy! Continue regular care."
        },
        "Tomato_Early_blight": {
            "severity": "Medium",
            "color": "🟡",
            "treatment": (
                "Use fungicide spray. Remove lower infected "
                "leaves. Mulch around base of plant."
            )
        },
        "Tomato_Late_blight": {
            "severity": "High",
            "color": "🔴",
            "treatment": (
                "Apply fungicide immediately. Remove and destroy "
                "infected plants. Improve air circulation."
            )
        },
        "Tomato_healthy": {
            "severity": "None",
            "color": "🟢",
            "treatment": "Plant is healthy! Continue regular care."
        }
    }
    return info.get(class_name, {
        "severity": "Unknown",
        "color": "⚪",
        "treatment": "Consult an agricultural expert."
    })


# ─────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/plant-under-sun.png",
             width=80)
    st.title("🌿 Plant Doctor")
    st.markdown("---")

    # API Health
    health = check_api_health()
    if health:
        st.success("✅ API Connected")
        st.caption(
            f"Model: PlantDiseaseDetector@champion")
        st.caption(f"Classes: {health['num_classes']}")
        st.caption(f"Container: {health['container_id']}")
    else:
        st.error("❌ API Offline")
        st.caption("Start the FastAPI backend first")

    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["🔍 Single Detection",
         "📦 Bulk Detection",
         "📊 Pipeline Console",
         "ℹ️ About"]
    )
    st.markdown("---")
    st.caption("DA5402 — Final Project")
    st.caption("Plant Disease Detection")

# ─────────────────────────────────────────
# Page: Single Detection
# ─────────────────────────────────────────
if page == "🔍 Single Detection":
    st.markdown(
        '<div class="main-header">🌿 Plant Disease Detector'
        '</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">'
        'Upload a leaf image to detect diseases instantly'
        '</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        uploaded = st.file_uploader(
            "Upload Leaf Image",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear image of a plant leaf"
        )

        if uploaded:
            image = Image.open(uploaded)
            st.image(image, caption="Uploaded Image",
                     use_container_width=True)

    with col2:
        if uploaded:
            with st.spinner("Analyzing leaf..."):
                img_bytes = uploaded.getvalue()
                result = predict_single(
                    img_bytes, uploaded.name)

            if result and "error" not in result:
                disease_info = get_disease_info(
                    result["predicted_class"])

                st.markdown("### 🔬 Analysis Result")
                st.markdown(
                    f'<div class="result-box">'
                    f'<h3>{disease_info["color"]} '
                    f'{result["predicted_class"]}</h3>'
                    f'<p><b>Confidence:</b> '
                    f'{result["confidence"]*100:.1f}%</p>'
                    f'<p><b>Severity:</b> '
                    f'{disease_info["severity"]}</p>'
                    f'<p><b>Treatment:</b> '
                    f'{disease_info["treatment"]}</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )

                st.markdown("### 📊 Class Probabilities")
                probs = result["all_probabilities"]
                for cls, prob in sorted(
                        probs.items(),
                        key=lambda x: x[1],
                        reverse=True):
                    st.progress(
                        prob,
                        text=f"{cls}: {prob*100:.1f}%"
                    )

                st.markdown("### ⚡ Performance")
                m1, m2 = st.columns(2)
                m1.metric("Inference Time",
                          f"{result['inference_time_ms']:.0f}ms")
                m2.metric("Confidence",
                          f"{result['confidence']*100:.1f}%")

            elif result and "error" in result:
                st.markdown(
                    f'<div class="error-box">'
                    f'❌ Error: {result["error"]}'
                    f'</div>',
                    unsafe_allow_html=True
                )
        else:
            st.info(
                "👆 Upload a leaf image to get started\n\n"
                "Supported plants:\n"
                "- 🥔 Potato (Early Blight, Late Blight, Healthy)\n"
                "- 🍅 Tomato (Early Blight, Late Blight, Healthy)"
            )

# ─────────────────────────────────────────
# Page: Bulk Detection
# ─────────────────────────────────────────
elif page == "📦 Bulk Detection":
    st.markdown(
        '<div class="main-header">📦 Bulk Detection'
        '</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">'
        'Upload multiple images or a ZIP file for batch analysis'
        '</div>', unsafe_allow_html=True)

    upload_type = st.radio(
        "Upload Type",
        ["Multiple Images", "ZIP File"],
        horizontal=True
    )

    files_to_process = []

    if upload_type == "Multiple Images":
        uploaded_files = st.file_uploader(
            "Upload Multiple Images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True
        )
        if uploaded_files:
            files_to_process = [
                (f.name, f.getvalue())
                for f in uploaded_files
            ]

    else:
        zip_file = st.file_uploader(
            "Upload ZIP File",
            type=["zip"]
        )
        if zip_file:
            with zipfile.ZipFile(
                    io.BytesIO(zip_file.getvalue())) as zf:
                for name in zf.namelist():
                    if name.lower().endswith(
                            ('.jpg', '.jpeg', '.png')):
                        files_to_process.append(
                            (name, zf.read(name)))
            st.info(
                f"Found {len(files_to_process)} "
                f"images in ZIP")

    if files_to_process:
        st.info(f"Ready to process "
                f"{len(files_to_process)} images")

        if st.button("🚀 Run Batch Analysis",
                     type="primary"):
            with st.spinner(
                    f"Processing "
                    f"{len(files_to_process)} images..."):
                result = predict_batch(files_to_process)

            if result and "error" not in result:
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Images",
                          result["total_images"])
                c2.metric("Successful",
                          result["successful"])
                c3.metric("Failed", result["failed"])

                st.markdown("### 📋 Results")
                for r in result["results"]:
                    if r["status"] == "success":
                        info = get_disease_info(
                            r["predicted_class"])
                        with st.expander(
                                f"{info['color']} "
                                f"{r['filename']}"):
                            st.write(
                                f"**Disease:** "
                                f"{r['predicted_class']}")
                            st.write(
                                f"**Confidence:** "
                                f"{r['confidence']*100:.1f}%")
                            st.write(
                                f"**Severity:** "
                                f"{info['severity']}")
                    else:
                        st.error(
                            f"❌ {r['filename']}: "
                            f"{r.get('error', 'Failed')}")

# ─────────────────────────────────────────
# Page: Pipeline Console
# ─────────────────────────────────────────
elif page == "📊 Pipeline Console":
    st.markdown(
        '<div class="main-header">📊 Pipeline Console'
        '</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([
        "🔬 Model Registry",
        "📈 Live Metrics",
        "🔗 Tool Links"
    ])

    with tab1:
        st.markdown("### 🏆 Model Performance Comparison")
        import json
        try:
            with open("metrics/final_metrics.json") as f:
                final = json.load(f)

            st.success(
                f"Best Model: **{final['best_model']}** "
                f"— Test F1: **{final['test_macro_f1']:.4f}**"
            )

            models = final.get("all_models", [])
            if models:
                import pandas as pd
                df = pd.DataFrame(models)
                df["f1"] = df["f1"].round(4)
                df["accuracy"] = df["accuracy"].round(4)
                df.columns = [
                    "Model", "Macro F1", "Accuracy"]
                st.dataframe(df, use_container_width=True)

            st.markdown("### 📊 Per-Class F1 Scores")
            report = final.get(
                "classification_report", {})
            class_scores = {
                k: v for k, v in report.items()
                if isinstance(v, dict) and
                "f1-score" in v
            }
            for cls, scores in class_scores.items():
                if cls not in [
                        "macro avg", "weighted avg"]:
                    st.progress(
                        scores["f1-score"],
                        text=(f"{cls}: "
                              f"F1={scores['f1-score']:.3f} "
                              f"P={scores['precision']:.3f} "
                              f"R={scores['recall']:.3f}")
                    )
        except FileNotFoundError:
            st.warning(
                "Run src/evaluate.py first to see metrics")

    with tab2:
        st.markdown("### 📡 Live API Metrics")
        if st.button("🔄 Refresh Metrics"):
            try:
                r = requests.get(
                    f"{API_BASE}/metrics", timeout=5)
                metrics_text = r.text
                lines = [
                    l for l in metrics_text.split("\n")
                    if not l.startswith("#") and l.strip()
                ]
                st.code(
                    "\n".join(lines[:30]),
                    language="text"
                )
            except Exception:
                st.error("Could not fetch metrics")

        st.markdown("### 🏥 API Health")
        health = check_api_health()
        if health:
            col1, col2 = st.columns(2)
            col1.metric("Status", health["status"])
            col2.metric("Classes", health["num_classes"])
            st.json(health)

    with tab3:
        st.markdown("### 🔗 MLOps Tool Links")
        st.markdown("""
        | Tool | URL | Purpose |
        |------|-----|---------|
        | MLflow UI | [127.0.0.1:5000](http://127.0.0.1:5000) | Experiment tracking |
        | FastAPI Docs | [127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) | API documentation |
        | Prometheus | [127.0.0.1:9090](http://127.0.0.1:9090) | Metrics collection |
        | Grafana | [127.0.0.1:3000](http://127.0.0.1:3000) | Monitoring dashboard |
        | Airflow | [127.0.0.1:8080](http://127.0.0.1:8080) | Pipeline orchestration |
        | DagsHub | [dagshub.com/ajazhsn/Final_project](https://dagshub.com/ajazhsn/Final_project) | Data versioning |
        """)

# ─────────────────────────────────────────
# Page: About
# ─────────────────────────────────────────
elif page == "ℹ️ About":
    st.markdown(
        '<div class="main-header">ℹ️ About'
        '</div>', unsafe_allow_html=True)

    st.markdown("""
    ## Plant Disease Detection System

    This application uses deep learning to identify plant
    diseases from leaf images.

    ### 🏗️ Architecture
    - **Frontend:** Streamlit (this app)
    - **Backend:** FastAPI REST API
    - **Model:** Custom CNN (3-block) fine-tuned
    - **Tracking:** MLflow experiment tracking
    - **Versioning:** DVC + DagsHub
    - **Orchestration:** Apache Airflow
    - **Monitoring:** Prometheus + Grafana

    ### 📊 Model Performance
    | Model | Test F1 | Test Accuracy |
    |-------|---------|---------------|
    | SVM + HOG | 0.6132 | 0.6792 |
    | MLP | 0.5843 | 0.6205 |
    | **Custom CNN (Champion)** | **0.9083** | **0.9157** |

    ### 🌿 Supported Classes
    - 🥔 Potato Early Blight
    - 🥔 Potato Late Blight
    - 🥔 Potato Healthy
    - 🍅 Tomato Early Blight
    - 🍅 Tomato Late Blight
    - 🍅 Tomato Healthy

    ### 📚 Course
    DA5402 — Graduate Machine Learning Operations (MLOps)
    """)
