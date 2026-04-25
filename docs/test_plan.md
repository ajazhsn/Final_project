# Test Plan & Test Report
## Plant Disease Detection — DA5402

---

## 1. Test Plan

### 1.1 Scope
Unit testing of all FastAPI REST API endpoints for the Plant Disease Detection system.

### 1.2 Acceptance Criteria
- All endpoints return correct HTTP status codes
- Prediction accuracy > 80% confidence on known test images
- Probabilities sum to 1.0 (±0.01)
- Prometheus metrics endpoint exposes required metric names
- Invalid inputs return appropriate error codes

### 1.3 Test Environment
- Python 3.12
- pytest 9.0.3
- FastAPI running in Docker container
- Test runner: `python3 -m pytest src/test_api.py -v`

---

## 2. Test Cases

### Module: Health Endpoint
| ID | Test Case | Expected | Status |
|----|-----------|----------|--------|
| TC-01 | GET /health returns 200 | HTTP 200 | ✅ PASS |
| TC-02 | Health status = "healthy" | status: healthy | ✅ PASS |
| TC-03 | Model loaded = true | model_loaded: true | ✅ PASS |
| TC-04 | Response has container_id | container_id present | ✅ PASS |
| TC-05 | num_classes = 6 | num_classes: 6 | ✅ PASS |

### Module: Predict Endpoint
| ID | Test Case | Expected | Status |
|----|-----------|----------|--------|
| TC-06 | POST /predict returns 200 | HTTP 200 | ✅ PASS |
| TC-07 | Correct class predicted | Potato___Early_blight | ✅ PASS |
| TC-08 | Confidence > 0.8 | confidence > 0.8 | ✅ PASS |
| TC-09 | Returns 6 probabilities | len(probs) == 6 | ✅ PASS |
| TC-10 | Probabilities sum to 1.0 | sum ≈ 1.0 | ✅ PASS |
| TC-11 | inference_time_ms > 0 | time > 0 | ✅ PASS |
| TC-12 | Invalid file returns 400 | HTTP 400 | ✅ PASS |

### Module: Classes Endpoint
| ID | Test Case | Expected | Status |
|----|-----------|----------|--------|
| TC-13 | GET /classes returns 200 | HTTP 200 | ✅ PASS |
| TC-14 | Returns 6 classes | num_classes: 6 | ✅ PASS |
| TC-15 | Contains Potato classes | Potato in classes | ✅ PASS |
| TC-16 | Contains Tomato classes | Tomato in classes | ✅ PASS |

### Module: Metrics Endpoint
| ID | Test Case | Expected | Status |
|----|-----------|----------|--------|
| TC-17 | GET /metrics returns 200 | HTTP 200 | ✅ PASS |
| TC-18 | Contains images counter | plant_images_processed_total | ✅ PASS |
| TC-19 | Contains latency histogram | plant_inference_latency_seconds | ✅ PASS |

---

## 3. Test Report

**Date:** April 2026  
**Tester:** Ajaz Hussain  
**Environment:** Docker container, Python 3.12, pytest 9.0.3

========================================================= test session starts
platform linux -- Python 3.12.3, pytest-9.0.3, pluggy-1.6.0
collected 19 items
src/test_api.py::TestHealthEndpoint::test_health_returns_200         PASSED
src/test_api.py::TestHealthEndpoint::test_health_status_healthy       PASSED
src/test_api.py::TestHealthEndpoint::test_health_model_loaded         PASSED
src/test_api.py::TestHealthEndpoint::test_health_has_container_id     PASSED
src/test_api.py::TestHealthEndpoint::test_health_correct_num_classes  PASSED
src/test_api.py::TestPredictEndpoint::test_predict_returns_200        PASSED
src/test_api.py::TestPredictEndpoint::test_predict_correct_class      PASSED
src/test_api.py::TestPredictEndpoint::test_predict_high_confidence    PASSED
src/test_api.py::TestPredictEndpoint::test_predict_has_all_probs      PASSED
src/test_api.py::TestPredictEndpoint::test_predict_probs_sum_to_one   PASSED
src/test_api.py::TestPredictEndpoint::test_predict_has_inference_time PASSED
src/test_api.py::TestPredictEndpoint::test_predict_invalid_file_400   PASSED
src/test_api.py::TestClassesEndpoint::test_classes_returns_200        PASSED
src/test_api.py::TestClassesEndpoint::test_classes_has_6_classes      PASSED
src/test_api.py::TestClassesEndpoint::test_classes_contains_potato    PASSED
src/test_api.py::TestClassesEndpoint::test_classes_contains_tomato    PASSED
src/test_api.py::TestMetricsEndpoint::test_metrics_returns_200        PASSED
src/test_api.py::TestMetricsEndpoint::test_metrics_contains_counter   PASSED
src/test_api.py::TestMetricsEndpoint::test_metrics_contains_histogram PASSED
========================================================== 19 passed in 1.41s

**Result: 19/19 PASSED — Acceptance criteria met ✅**

