# 🌿 Plant Disease Detection — End-to-End MLOps Application

**Course:** DA5402 — Graduate Machine Learning Operations  
**Student:** Ajaz Hussain (DA25M006)  
**DagsHub:** [ajazhsn/Final_project](https://dagshub.com/ajazhsn/Final_project)

## Model Performance
| Model | Test F1 | Test Accuracy |
|-------|---------|---------------|
| SVM + HOG | 0.6132 | 0.6792 |
| MLP | 0.5843 | 0.6205 |
| **Custom CNN (Champion)** | **0.9083** | **0.9157** |

## Quick Start
```bash
# Start all services
docker compose up -d --scale backend=3
mlflow server --host 127.0.0.1 --port 5000
streamlit run app.py --server.port 8501
```

## Architecture
- **Frontend:** Streamlit (port 8501)
- **Backend:** FastAPI + Docker (ports 8000-8002)
- **Model:** Custom CNN — PlantDiseaseDetector@champion
- **Tracking:** MLflow (port 5000)
- **Orchestration:** Apache Airflow (port 8080)
- **Monitoring:** Prometheus + Grafana (ports 9090, 3000)
- **Data Versioning:** DVC + DagsHub