# Low-Level Design (LLD)
## Plant Disease Detection — API Endpoint Specifications

---

## 1. API Overview

**Base URL:** `http://localhost:8000`  
**Protocol:** HTTP/1.1  
**Content-Type:** `application/json` (responses), `multipart/form-data` (uploads)  
**Authentication:** None (internal service)

---

## 2. Endpoint Specifications

### 2.1 GET /

**Description:** Root endpoint — API information  
**Request:** None  
**Response (200):**
```json
{
  "message": "Plant Disease Detection API",
  "docs": "/docs",
  "health": "/health",
  "metrics": "/metrics"
}
```

---

### 2.2 GET /health

**Description:** Service health check  
**Request:** None  
**Response (200):**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "num_classes": 6,
  "classes": [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_healthy"
  ],
  "container_id": "b5fec554a942",
  "timestamp": "2026-04-22T10:00:00.000000"
}
```

---

### 2.3 POST /predict

**Description:** Single image plant disease prediction  
**Request:**
- Content-Type: `multipart/form-data`
- Body: `file` (image/jpeg, image/png)

**Response (200):**
```json
{
  "filename": "leaf_image.jpg",
  "predicted_class": "Potato___Early_blight",
  "confidence": 0.9868,
  "all_probabilities": {
    "Potato___Early_blight": 0.9868,
    "Potato___Late_blight": 0.0018,
    "Potato___healthy": 0.0,
    "Tomato_Early_blight": 0.0045,
    "Tomato_Late_blight": 0.007,
    "Tomato_healthy": 0.0
  },
  "inference_time_ms": 301.54,
  "model_version": "PlantDiseaseDetector@champion",
  "container_id": "b5fec554a942"
}
```

**Response (400):** File is not an image  
**Response (500):** Inference failed

---

### 2.4 POST /predict/batch

**Description:** Bulk image prediction  
**Request:**
- Content-Type: `multipart/form-data`
- Body: `files` (list of images)

**Response (200):**
```json
{
  "total_images": 3,
  "successful": 3,
  "failed": 0,
  "total_time_ms": 450.23,
  "container_id": "b5fec554a942",
  "results": [
    {
      "filename": "image1.jpg",
      "predicted_class": "Tomato_Late_blight",
      "confidence": 0.9993,
      "all_probabilities": {},
      "status": "success"
    }
  ]
}
```

---

### 2.5 GET /metrics

**Description:** Prometheus metrics endpoint  
**Request:** None  
**Response (200):** Plain text Prometheus format

**Metrics exposed:**
| Metric | Type | Description |
|--------|------|-------------|
| `plant_images_processed_total` | Counter | Total images by mode and class |
| `plant_inference_latency_seconds` | Histogram | Inference time distribution |
| `plant_active_requests` | Gauge | Current active requests |
| `plant_model_load_time_seconds` | Gauge | Model load time |
| `plant_prediction_confidence` | Histogram | Confidence score distribution |
| `plant_prediction_errors_total` | Counter | Errors by type |
| `plant_bulk_batch_size` | Histogram | Batch size distribution |

---

### 2.6 GET /classes

**Description:** List supported disease classes  
**Request:** None  
**Response (200):**
```json
{
  "num_classes": 6,
  "classes": [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_healthy"
  ],
  "model": "PlantDiseaseDetector@champion"
}
```

---

## 3. Model Architecture

### Custom CNN (PlantDiseaseDetector@champion)

Input: RGB image (224 × 224 × 3)
↓
Conv Block 1: Conv2d(3→32, 3×3) → BatchNorm → ReLU → MaxPool(2×2)
↓
Conv Block 2: Conv2d(32→64, 3×3) → BatchNorm → ReLU → MaxPool(2×2)
↓
Conv Block 3: Conv2d(64→128, 3×3) → BatchNorm → ReLU → MaxPool(2×2)
↓
AdaptiveAvgPool2d(4×4) → Flatten
↓
FC: 2048 → 256 → ReLU → Dropout(0.3)
↓
Output: 256 → 6 (Softmax)

**Training configuration:**
- Optimizer: Adam (lr=0.0005)
- Loss: CrossEntropyLoss
- Epochs: 25
- Batch size: 32
- Augmentation: RandomHorizontalFlip, RandomRotation(15°), ColorJitter

---

## 4. DVC Pipeline Stages

```yaml
prepare:  data/raw → data/processed/v1
train_svm: HOG features + SVM classifier
train_mlp: Flattened pixels + MLP
train_cnn: Custom CNN (champion)
evaluate:  Test set evaluation → metrics/final_metrics.json
```

---

## 5. Airflow DAG Tasks

| Task | Operator | Pool | Description |
|------|----------|------|-------------|
| start | EmptyOperator | - | Pipeline start |
| sense_csv_file | FileSensor | default | Watch for CSV |
| init_db_task | PythonOperator | default | Initialize SQLite |
| read_csv_task | PythonOperator | scraper_pool | Read image paths |
| batch_predict_task | PythonOperator | scraper_pool | Run predictions |
| check_failures_task | PythonOperator | default | Check errors |
| db_stats_task | PythonOperator | default | DB statistics |
| notify_batch_complete | EmailOperator | - | Success email |
| notify_failures | EmailOperator | - | Failure email |
| archive_csv_task | PythonOperator | default | Archive CSV |
| end | EmptyOperator | - | Pipeline end |