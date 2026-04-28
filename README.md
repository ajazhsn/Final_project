# 🌿 Plant Disease Detection — End-to-End MLOps Application

**Course:** DA5402 — Graduate Machine Learning Operations  
**Student:** Ajaz Hussain (DA25M006)  
**GitHub:** [ajazhsn/Final_project](https://github.com/ajazhsn/Final_project)  
**DagsHub:** [ajazhsn/Final_project](https://dagshub.com/ajazhsn/Final_project)

---

## 🚀 Quick Start (One Command)

```bash
# Clone the repository
git clone https://github.com/ajazhsn/Final_project.git
cd Final_project

# Pull data from DagsHub
dvc pull

# Build the Docker image
docker build -t plant-disease-api:latest .

# Start ALL services
docker compose up -d --scale backend=3
```

**All services will be available at:**
| Service | URL | Credentials |
|---------|-----|-------------|
| 🌿 Streamlit App | http://localhost:8501 | - |
| ⚡ FastAPI Docs | http://localhost:8000/docs | - |
| 🔬 MLflow UI | http://localhost:5000 | - |
| 📊 Grafana | http://localhost:3000 | admin/admin123 |
| 📡 Prometheus | http://localhost:9090 | - |
| 🔄 Airflow | http://localhost:8080 | admin/admin123 |

---

## 📊 Model Performance

| Model | Test F1 | Test Accuracy | Train Time |
|-------|---------|---------------|------------|
| SVM + HOG | 0.6132 | 0.6792 | 37s |
| MLP | 0.5843 | 0.6205 | 3.2 min |
| **Custom CNN (Champion)** | **0.9083** | **0.9157** | 13.4 min |

---

## 🏗️ Architecture

Streamlit (8501) → FastAPI x3 (8000-8002) → CNN Model
↓
MLflow (5000) ← Experiment Tracking    Prometheus (9090)
DVC + DagsHub ← Data Versioning        Grafana (3000)
Airflow (8080) ← Orchestration         AlertManager (9093)

---

## 📁 Project Structure

├── src/
│   ├── data_prep.py      # Data pipeline
│   ├── transform.py      # Augmentation
│   ├── train.py          # 3 models (SVM, MLP, CNN)
│   ├── finetune.py       # Fine-tune from registry
│   ├── evaluate.py       # Evaluation + metrics
│   ├── inference.py      # FastAPI backend
│   ├── monitor.py        # Data drift monitoring
│   └── test_api.py       # 19/19 unit tests
├── airflow/dags/         # Airflow pipeline
├── monitoring/           # Prometheus + Grafana
├── docs/                 # HLD, LLD, test plan, user manual
├── app.py                # Streamlit frontend
├── Dockerfile            # Container definition
├── docker-compose.yml    # Full stack orchestration
├── MLproject             # MLflow project
├── dvc.yaml              # DVC pipeline
└── params.yaml           # Experiment parameters

---

## 🧪 Running Tests

```bash
pip install pytest
python3 -m pytest src/test_api.py -v
# Result: 19/19 PASSED
```

---

## 🔄 DVC Pipeline

```bash
dvc repro          # Run full pipeline
dvc metrics show   # Show all metrics
dvc plots diff     # Compare experiments
```

---

## 📚 Documentation

- [High-Level Design](docs/HLD.md)
- [Low-Level Design](docs/LLD.md)
- [Test Plan & Report](docs/test_plan.md)
- [User Manual](docs/user_manual.md)

