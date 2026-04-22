# src/test_api.py — Unit tests for Plant Disease Detection API
import io
import json
import pytest
import requests
from pathlib import Path

API_BASE = "http://127.0.0.1:8000"
TEST_IMAGE = "data/raw/Potato___Early_blight/001187a0-57ab-4329-baff-e7246a9edeb0___RS_Early.B 8178.JPG"


class TestHealthEndpoint:
    def test_health_returns_200(self):
        r = requests.get(f"{API_BASE}/health")
        assert r.status_code == 200

    def test_health_status_healthy(self):
        r = requests.get(f"{API_BASE}/health")
        assert r.json()["status"] == "healthy"

    def test_health_model_loaded(self):
        r = requests.get(f"{API_BASE}/health")
        assert r.json()["model_loaded"] is True

    def test_health_has_container_id(self):
        r = requests.get(f"{API_BASE}/health")
        assert "container_id" in r.json()

    def test_health_correct_num_classes(self):
        r = requests.get(f"{API_BASE}/health")
        assert r.json()["num_classes"] == 6


class TestPredictEndpoint:
    def test_predict_returns_200(self):
        with open(TEST_IMAGE, "rb") as f:
            r = requests.post(
                f"{API_BASE}/predict",
                files={"file": ("test.jpg", f, "image/jpeg")}
            )
        assert r.status_code == 200

    def test_predict_correct_class(self):
        with open(TEST_IMAGE, "rb") as f:
            r = requests.post(
                f"{API_BASE}/predict",
                files={"file": ("test.jpg", f, "image/jpeg")}
            )
        result = r.json()
        assert result["predicted_class"] == "Potato___Early_blight"

    def test_predict_high_confidence(self):
        with open(TEST_IMAGE, "rb") as f:
            r = requests.post(
                f"{API_BASE}/predict",
                files={"file": ("test.jpg", f, "image/jpeg")}
            )
        assert r.json()["confidence"] > 0.8

    def test_predict_has_all_probabilities(self):
        with open(TEST_IMAGE, "rb") as f:
            r = requests.post(
                f"{API_BASE}/predict",
                files={"file": ("test.jpg", f, "image/jpeg")}
            )
        probs = r.json()["all_probabilities"]
        assert len(probs) == 6

    def test_predict_probabilities_sum_to_one(self):
        with open(TEST_IMAGE, "rb") as f:
            r = requests.post(
                f"{API_BASE}/predict",
                files={"file": ("test.jpg", f, "image/jpeg")}
            )
        probs = r.json()["all_probabilities"]
        assert abs(sum(probs.values()) - 1.0) < 0.01

    def test_predict_has_inference_time(self):
        with open(TEST_IMAGE, "rb") as f:
            r = requests.post(
                f"{API_BASE}/predict",
                files={"file": ("test.jpg", f, "image/jpeg")}
            )
        assert r.json()["inference_time_ms"] > 0

    def test_predict_invalid_file_returns_400(self):
        r = requests.post(
            f"{API_BASE}/predict",
            files={"file": ("test.txt", b"not an image",
                            "text/plain")}
        )
        assert r.status_code == 400


class TestClassesEndpoint:
    def test_classes_returns_200(self):
        r = requests.get(f"{API_BASE}/classes")
        assert r.status_code == 200

    def test_classes_has_6_classes(self):
        r = requests.get(f"{API_BASE}/classes")
        assert r.json()["num_classes"] == 6

    def test_classes_contains_potato(self):
        r = requests.get(f"{API_BASE}/classes")
        classes = r.json()["classes"]
        assert any("Potato" in c for c in classes)

    def test_classes_contains_tomato(self):
        r = requests.get(f"{API_BASE}/classes")
        classes = r.json()["classes"]
        assert any("Tomato" in c for c in classes)


class TestMetricsEndpoint:
    def test_metrics_returns_200(self):
        r = requests.get(f"{API_BASE}/metrics")
        assert r.status_code == 200

    def test_metrics_contains_plant_counter(self):
        r = requests.get(f"{API_BASE}/metrics")
        assert "plant_images_processed_total" in r.text

    def test_metrics_contains_latency_histogram(self):
        r = requests.get(f"{API_BASE}/metrics")
        assert "plant_inference_latency_seconds" in r.text


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
