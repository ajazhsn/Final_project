# High-Level Design (HLD)
## Plant Disease Detection — End-to-End MLOps Application

**Course:** DA5402 — Graduate Machine Learning Operations  
**Student:** Ajaz Hussain (DA25M006)  
**Version:** 1.0

---

## 1. System Overview

The Plant Disease Detection system is a production-grade MLOps application that identifies plant diseases from leaf images. It covers the complete ML lifecycle — from data ingestion and model training to serving, monitoring, and orchestration.

---

## 2. Architecture Diagram

┌─────────────────────────────────────────────────────────────┐
│                    USER LAYER                                │
│  ┌─────────────────────┐    ┌──────────────────────────┐   │
│  │  Streamlit Frontend  │    │   FastAPI Swagger UI      │   │
│  │  (port 8501)        │    │   (port 8000/docs)        │   │
│  └──────────┬──────────┘    └──────────────────────────┘   │
└─────────────┼───────────────────────────────────────────────┘
│ REST API calls
┌─────────────▼───────────────────────────────────────────────┐
│                  INFERENCE LAYER                             │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐        │
│  │  FastAPI     │ │  FastAPI     │ │  FastAPI     │        │
│  │  Container 1 │ │  Container 2 │ │  Container 3 │        │
│  │  port 8000   │ │  port 8001   │ │  port 8002   │        │
│  └──────────────┘ └──────────────┘ └──────────────┘        │
│         Docker Compose — 3 replicas                         │
└─────────────────────────────────────────────────────────────┘
│
┌─────────────▼───────────────────────────────────────────────┐
│                   MODEL LAYER                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  MLflow Model Registry                               │   │
│  │  PlantDiseaseDetector@champion (Custom CNN v2)       │   │
│  │  Test F1: 0.9083 | Test Accuracy: 0.9157            │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
│
┌─────────────▼───────────────────────────────────────────────┐
│              DATA & EXPERIMENT LAYER                         │
│  ┌──────────────────┐    ┌──────────────────────────────┐  │
│  │  DVC + DagsHub   │    │  MLflow Experiment Tracking  │  │
│  │  Data versioning │    │  Parameters, Metrics,        │  │
│  │  data-v0, v1, v2 │    │  Artifacts, Models           │  │
│  └──────────────────┘    └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
│
┌─────────────▼───────────────────────────────────────────────┐
│              ORCHESTRATION LAYER                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Apache Airflow (port 8080)                          │   │
│  │  FileSensor → predict → SQLite DB → notify          │   │
│  │  Pool: scraper_pool (3 workers)                      │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
│
┌─────────────▼───────────────────────────────────────────────┐
│              MONITORING LAYER                                │
│  ┌────────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │  Prometheus    │  │  Grafana     │  │  node_exporter │  │
│  │  port 9090     │  │  port 3000   │  │  port 9100     │  │
│  │  Scrapes API   │  │  Dashboard   │  │  System metrics│  │
│  └────────────────┘  └──────────────┘  └────────────────┘  │
└─────────────────────────────────────────────────────────────┘
---

## 3. Component Description

### 3.1 Frontend (Streamlit)
- Single image upload and prediction display
- Bulk ZIP upload for batch processing
- Pipeline console showing MLflow + Grafana links
- Disease severity and treatment recommendations
- Loose coupling via REST API calls to backend

### 3.2 Backend (FastAPI)
- `/predict` — Single image inference
- `/predict/batch` — Bulk image inference
- `/health` — Service health check
- `/metrics` — Prometheus metrics endpoint
- `/classes` — Supported disease classes
- Containerized with Docker, 3 replicas for load distribution

### 3.3 Model Registry (MLflow)
- Experiment tracking for SVM, MLP, and CNN models
- Model registration and versioning
- Champion model: PlantDiseaseDetector@champion (v2)
- MLproject entry points for reproducible training

### 3.4 Data Versioning (DVC + DagsHub)
- Raw dataset tracked as data-v0-raw
- Processed dataset tracked as data-v1
- Full DVC pipeline: prepare → train → evaluate
- Remote storage on DagsHub

### 3.5 Orchestration (Airflow)
- FileSensor monitors input directory for CSV files
- Batch prediction pipeline with 3-worker pool
- Results stored in SQLite database
- Email alerts for failures and batch statistics

### 3.6 Monitoring (Prometheus + Grafana)
- Custom metrics: Counters, Gauges, Histograms, Summaries
- System metrics via node_exporter
- Real-time Grafana dashboard with 10 panels
- AlertManager rules for CPU, latency, error rate

---

## 4. Design Decisions

### 4.1 Why Custom CNN over EfficientNet?
CPU-only training environment made large pretrained models impractical. A 3-block custom CNN trained in ~15 minutes on CPU achieved F1 > 0.90, meeting the project requirements efficiently.

### 4.2 Why DagsHub over plain S3?
DagsHub provides integrated Git + DVC remote in one platform, with built-in experiment tracking UI and MLflow integration — reducing infrastructure complexity.

### 4.3 Why Docker Compose over Swarm?
Hardware constraint (7.6GB RAM, single machine) made running two simultaneous VMs for Swarm impractical. Docker Compose with 3 replicas demonstrates equivalent orchestration concepts.

### 4.4 Why SQLite for Airflow pipeline DB?
Lightweight, zero-configuration database suitable for development and demonstration. In production, PostgreSQL would be used.

### 4.5 Loose Coupling Principle
Frontend and backend are independent services connected only via REST API. The Streamlit app never directly imports model code — it always goes through the FastAPI endpoint.

---

## 5. Technology Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| Frontend | Streamlit | 1.56.0 |
| Backend | FastAPI + Uvicorn | 0.136.0 |
| ML Framework | PyTorch | 2.11.0 |
| Experiment Tracking | MLflow | 3.11.1 |
| Data Versioning | DVC | 3.67.1 |
| Containerization | Docker Compose | v5.0.2 |
| Orchestration | Apache Airflow | 2.9.3 |
| Monitoring | Prometheus | 2.51.0 |
| Dashboard | Grafana | 10.4.2 |
| Language | Python | 3.12 |


