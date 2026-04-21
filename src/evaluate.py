# src/evaluate.py
import os
import json
import yaml
import logging
import numpy as np
from pathlib import Path
from datetime import datetime

import mlflow
from sklearn.metrics import (
    f1_score, accuracy_score,
    confusion_matrix, classification_report
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[
        logging.FileHandler("logs/evaluate.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def load_test_data(v1_dir, class_names, image_size):
    """Load test set images."""
    from PIL import Image
    X, y = [], []
    test_dir = v1_dir / "test"
    for label_idx, class_name in enumerate(class_names):
        class_dir = test_dir / class_name
        if not class_dir.exists():
            continue
        images = (list(class_dir.glob("*.jpg")) +
                  list(class_dir.glob("*.JPG")))
        for img_path in images:
            try:
                with Image.open(img_path) as img:
                    img = img.convert("RGB").resize(
                        (image_size, image_size))
                    X.append(np.array(img))
                    y.append(label_idx)
            except Exception as e:
                logger.error(f"Failed: {img_path}: {e}")
        logger.info(f"  {class_name}: {len(images)} test images")
    return np.array(X), np.array(y)


def evaluate_svm(X_test, y_test, class_names):
    """Evaluate SVM on test set."""
    import joblib
    from skimage.feature import hog

    model_path = "models/svm_model.pkl"
    if not Path(model_path).exists():
        logger.warning("SVM model not found, skipping")
        return None

    pipeline = joblib.load(model_path)

    # Extract HOG features
    features = []
    for img in X_test:
        feat = hog(img, orientations=9,
                   pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2),
                   channel_axis=-1)
        features.append(feat)
    X_hog = np.array(features)

    y_pred = pipeline.predict(X_hog)
    f1 = f1_score(y_test, y_pred, average="macro")
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred).tolist()

    logger.info(f"SVM — Test F1: {f1:.4f} | Acc: {acc:.4f}")
    return {"model": "SVM", "f1": f1,
            "accuracy": acc, "confusion_matrix": cm,
            "predictions": y_pred.tolist()}


def evaluate_mlp(X_test, y_test, class_names):
    """Evaluate MLP on test set."""
    import joblib

    model_path = "models/mlp_model.pkl"
    if not Path(model_path).exists():
        logger.warning("MLP model not found, skipping")
        return None

    pipeline = joblib.load(model_path)
    image_size = X_test.shape[1]
    X_flat = X_test.reshape(len(X_test), -1) / 255.0

    y_pred = pipeline.predict(X_flat)
    f1 = f1_score(y_test, y_pred, average="macro")
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred).tolist()

    logger.info(f"MLP — Test F1: {f1:.4f} | Acc: {acc:.4f}")
    return {"model": "MLP", "f1": f1,
            "accuracy": acc, "confusion_matrix": cm,
            "predictions": y_pred.tolist()}


def evaluate_cnn(X_test, y_test, class_names, image_size):
    """Evaluate CNN on test set."""
    import torch
    import torch.nn as nn
    from torchvision import transforms
    from torch.utils.data import DataLoader, Dataset
    from PIL import Image

    model_path = "models/cnn_finetuned.pt"
    if not Path(model_path).exists():
        model_path = "models/cnn_model.pt"
    if not Path(model_path).exists():
        logger.warning("CNN model not found, skipping")
        return None

    checkpoint = torch.load(
        model_path, map_location="cpu", weights_only=True)
    num_filters = checkpoint.get("num_filters", [32, 64, 128])
    num_classes = len(class_names)

    class PlantCNN(nn.Module):
        def __init__(self, num_classes, num_filters):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, num_filters[0], 3, padding=1),
                nn.BatchNorm2d(num_filters[0]),
                nn.ReLU(), nn.MaxPool2d(2, 2),
                nn.Conv2d(num_filters[0], num_filters[1],
                          3, padding=1),
                nn.BatchNorm2d(num_filters[1]),
                nn.ReLU(), nn.MaxPool2d(2, 2),
                nn.Conv2d(num_filters[1], num_filters[2],
                          3, padding=1),
                nn.BatchNorm2d(num_filters[2]),
                nn.ReLU(), nn.MaxPool2d(2, 2),
            )
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
                nn.Linear(num_filters[2] * 16, 256),
                nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    model = PlantCNN(num_classes, num_filters)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])

    all_preds = []
    with torch.no_grad():
        for img_arr in X_test:
            img = Image.fromarray(img_arr)
            tensor = transform(img).unsqueeze(0)
            output = model(tensor)
            pred = torch.argmax(output, dim=1).item()
            all_preds.append(pred)

    y_pred = np.array(all_preds)
    f1 = f1_score(y_test, y_pred, average="macro")
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred).tolist()

    logger.info(f"CNN — Test F1: {f1:.4f} | Acc: {acc:.4f}")
    return {"model": "CNN_Finetuned", "f1": f1,
            "accuracy": acc, "confusion_matrix": cm,
            "predictions": y_pred.tolist()}


def save_plots(results, class_names):
    """Save confusion matrix and F1 scores for DVC plots."""
    os.makedirs("plots", exist_ok=True)

    # F1 scores plot data
    f1_data = [
        {"model": r["model"], "f1_score": r["f1"]}
        for r in results if r is not None
    ]
    with open("plots/f1_scores.json", "w") as f:
        json.dump(f1_data, f, indent=2)

    # Confusion matrix for best model (CNN)
    best = max(
        [r for r in results if r is not None],
        key=lambda x: x["f1"]
    )
    cm_data = []
    cm = best["confusion_matrix"]
    for i, true_class in enumerate(class_names):
        for j, pred_class in enumerate(class_names):
            cm_data.append({
                "actual": true_class,
                "predicted": pred_class,
                "count": cm[i][j]
            })
    with open("plots/confusion_matrix.json", "w") as f:
        json.dump(cm_data, f, indent=2)

    logger.info("Plots saved to plots/")


def main():
    config = load_config()
    params = load_params()

    mlflow.set_tracking_uri(
        config["deployment"]["mlflow_tracking_uri"])

    v1_dir = Path(config["data"]["processed_dir"]) / "v1"
    image_size = params["prepare"]["image_size"]

    with open(v1_dir / "class_names.json") as f:
        class_map = json.load(f)
    class_names = list(class_map.keys())

    logger.info("Loading test set...")
    X_test, y_test = load_test_data(
        v1_dir, class_names, image_size)
    logger.info(f"Test samples: {len(X_test)}")

    # Evaluate all models
    logger.info("="*50)
    logger.info("Evaluating all models on TEST SET")
    logger.info("="*50)

    results = []
    results.append(evaluate_svm(X_test, y_test, class_names))
    results.append(evaluate_mlp(X_test, y_test, class_names))
    results.append(
        evaluate_cnn(X_test, y_test, class_names, image_size))

    valid = [r for r in results if r is not None]
    best = max(valid, key=lambda x: x["f1"])

    # Classification report for best model
    best_preds = np.array(best["predictions"])
    report = classification_report(
        y_test, best_preds,
        target_names=class_names, output_dict=True)

    logger.info(f"\n{'='*50}")
    logger.info("FINAL TEST SET RESULTS")
    logger.info(f"{'='*50}")
    for r in valid:
        logger.info(
            f"{r['model']:20s} | "
            f"F1: {r['f1']:.4f} | "
            f"Acc: {r['accuracy']:.4f}")
    logger.info(f"\nBest Model: {best['model']}")
    logger.info(f"Best Test Macro F1: {best['f1']:.4f}")

    # Save final metrics
    final_metrics = {
        "best_model": best["model"],
        "test_macro_f1": best["f1"],
        "test_accuracy": best["accuracy"],
        "evaluated_at": datetime.now().isoformat(),
        "all_models": [
            {"model": r["model"],
             "f1": r["f1"],
             "accuracy": r["accuracy"]}
            for r in valid
        ],
        "classification_report": report
    }

    os.makedirs("metrics", exist_ok=True)
    with open("metrics/final_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    # Save plots
    save_plots(valid, class_names)

    # Log to MLflow
    with mlflow.start_run(run_name="Final_Evaluation"):
        mlflow.log_param("best_model", best["model"])
        mlflow.log_metric("test_macro_f1", best["f1"])
        mlflow.log_metric("test_accuracy", best["accuracy"])
        for r in valid:
            mlflow.log_metric(
                f"test_f1_{r['model'].lower()}", r["f1"])
        mlflow.log_artifact("metrics/final_metrics.json")
        mlflow.log_artifact("plots/confusion_matrix.json")
        mlflow.log_artifact("plots/f1_scores.json")

    logger.info("Evaluation complete.")
    return final_metrics


if __name__ == "__main__":
    main()
