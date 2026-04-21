# src/train.py
import os
import sys
import json
import yaml
import time
import logging
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

import mlflow
import mlflow.sklearn
import mlflow.pytorch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[
        logging.FileHandler("logs/train.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def load_dataset_subset(split_dir: Path, class_names: list,
                        max_per_class: int, image_size: int):
    """
    Load images as numpy arrays, capped at max_per_class per class.
    Returns X (numpy array) and y (list of int labels).
    """
    # Assisted by Claude AI for image loading boilerplate
    from PIL import Image
    import random

    X, y = [], []
    for label_idx, class_name in enumerate(class_names):
        class_dir = split_dir / class_name
        if not class_dir.exists():
            logger.warning(f"Class dir not found: {class_dir}")
            continue

        images = list(class_dir.glob("*.jpg")) + \
            list(class_dir.glob("*.JPG"))
        random.shuffle(images)
        images = images[:max_per_class]

        for img_path in images:
            try:
                with Image.open(img_path) as img:
                    img = img.convert("RGB").resize(
                        (image_size, image_size))
                    X.append(np.array(img))
                    y.append(label_idx)
            except Exception as e:
                logger.error(f"Failed to load {img_path}: {e}")

        logger.info(f"  {class_name}: {len(images)} images loaded")

    return np.array(X), np.array(y)


# ─────────────────────────────────────────
# MODEL A: SVM with HOG features
# ─────────────────────────────────────────
def train_svm(config: dict, params: dict) -> dict:
    """Train SVM with HOG feature extraction."""
    from skimage.feature import hog
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import f1_score, accuracy_score
    import joblib

    logger.info("="*50)
    logger.info("Training Model A: SVM + HOG Features")
    logger.info("="*50)

    p = params["train_svm"]
    prep = params["prepare"]
    v1_dir = Path(config["data"]["processed_dir"]) / "v1"
    image_size = prep["image_size"]
    max_per_class = prep["max_images_per_class"]

    with open(v1_dir / "class_names.json") as f:
        class_map = json.load(f)
    class_names = list(class_map.keys())

    # Load data
    logger.info("Loading training data...")
    X_train, y_train = load_dataset_subset(
        v1_dir / "train", class_names, max_per_class, image_size)
    X_val, y_val = load_dataset_subset(
        v1_dir / "val", class_names, max_per_class, image_size)

    # Extract HOG features
    logger.info("Extracting HOG features...")

    def extract_hog(X):
        features = []
        for img in X:
            feat = hog(
                img,
                orientations=params["train_svm"].get(
                    "hog_orientations", 9),
                pixels_per_cell=(
                    params["train_svm"].get("hog_pixels_per_cell", 8),
                    params["train_svm"].get("hog_pixels_per_cell", 8)),
                cells_per_block=(2, 2),
                channel_axis=-1
            )
            features.append(feat)
        return np.array(features)

    X_train_hog = extract_hog(X_train)
    X_val_hog = extract_hog(X_val)
    logger.info(f"HOG feature shape: {X_train_hog.shape}")

    # MLflow tracking
    mlflow.set_experiment(
        config["deployment"]["mlflow_experiment_name"])

    parent_run_id = os.environ.get("MLFLOW_RUN_ID")
    with mlflow.start_run(run_name="SVM_HOG",
                          run_id=parent_run_id if parent_run_id else None,
                          nested=True if parent_run_id else False):
        # Log params
        mlflow.log_param("model_type", "SVM")
        mlflow.log_param("feature_type", "HOG")
        mlflow.log_param("C", p["C"])
        mlflow.log_param("kernel", p["kernel"])
        mlflow.log_param("max_per_class", max_per_class)
        mlflow.log_param("image_size", image_size)
        mlflow.log_param("hog_orientations",
                         p.get("hog_orientations", 9))
        mlflow.log_param("num_classes", len(class_names))
        mlflow.log_param("train_samples", len(X_train))

        # Train
        logger.info("Training SVM...")
        start = time.time()
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(C=p["C"], kernel=p["kernel"],
                        random_state=p["random_seed"],
                        probability=True))
        ])
        pipeline.fit(X_train_hog, y_train)
        train_time = time.time() - start
        logger.info(f"Training time: {train_time:.1f}s")

        # Evaluate
        y_pred = pipeline.predict(X_val_hog)
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average="macro")

        # Log metrics
        mlflow.log_metric("val_accuracy", acc)
        mlflow.log_metric("val_macro_f1", f1)
        mlflow.log_metric("train_time_seconds", train_time)

        logger.info(f"Val Accuracy: {acc:.4f}")
        logger.info(f"Val Macro F1: {f1:.4f}")

        # Save model
        os.makedirs("models", exist_ok=True)
        model_path = "models/svm_model.pkl"
        joblib.dump(pipeline, model_path)
        mlflow.sklearn.log_model(pipeline, "svm_model")

        # Save metadata
        metadata = {
            "model_type": "SVM",
            "feature_type": "HOG",
            "train_date": datetime.now().isoformat(),
            "val_accuracy": acc,
            "val_macro_f1": f1,
            "train_time_seconds": train_time,
            "class_names": class_names,
            "mlflow_run_id": mlflow.active_run().info.run_id,
            "params": p
        }
        with open("models/svm_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        run_id = mlflow.active_run().info.run_id

    # Save DVC metrics
    os.makedirs("metrics", exist_ok=True)
    metrics = {
        "val_accuracy": acc,
        "val_macro_f1": f1,
        "train_time_seconds": train_time,
        "mlflow_run_id": run_id
    }
    with open("metrics/svm_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"SVM model saved to {model_path}")
    return metrics


# ─────────────────────────────────────────
# MODEL B: MLP
# ─────────────────────────────────────────
def train_mlp(config: dict, params: dict) -> dict:
    """Train MLP on flattened image features."""
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import f1_score, accuracy_score
    import joblib

    logger.info("="*50)
    logger.info("Training Model B: MLP")
    logger.info("="*50)

    p = params["train_mlp"]
    prep = params["prepare"]
    v1_dir = Path(config["data"]["processed_dir"]) / "v1"
    image_size = prep["image_size"]
    max_per_class = prep["max_images_per_class"]

    with open(v1_dir / "class_names.json") as f:
        class_map = json.load(f)
    class_names = list(class_map.keys())

    # Load data
    logger.info("Loading training data...")
    X_train, y_train = load_dataset_subset(
        v1_dir / "train", class_names, max_per_class, image_size)
    X_val, y_val = load_dataset_subset(
        v1_dir / "val", class_names, max_per_class, image_size)

    # Flatten images
    X_train_flat = X_train.reshape(len(X_train), -1) / 255.0
    X_val_flat = X_val.reshape(len(X_val), -1) / 255.0
    logger.info(f"Flattened feature shape: {X_train_flat.shape}")

    mlflow.set_experiment(
        config["deployment"]["mlflow_experiment_name"])

    parent_run_id = os.environ.get("MLFLOW_RUN_ID")
    with mlflow.start_run(run_name="MLP_Flatten",
                          run_id=parent_run_id if parent_run_id else None,
                          nested=True if parent_run_id else False):
        mlflow.log_param("model_type", "MLP")
        mlflow.log_param("feature_type", "flattened_pixels")
        mlflow.log_param("hidden_layers",
                         str(p["hidden_layers"]))
        mlflow.log_param("learning_rate", p["learning_rate"])
        mlflow.log_param("epochs", p["epochs"])
        mlflow.log_param("max_per_class", max_per_class)
        mlflow.log_param("train_samples", len(X_train))

        logger.info("Training MLP...")
        start = time.time()
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=tuple(p["hidden_layers"]),
                learning_rate_init=p["learning_rate"],
                max_iter=p["epochs"],
                random_state=p["random_seed"],
                verbose=True,
                early_stopping=True,
                validation_fraction=0.1
            ))
        ])
        pipeline.fit(X_train_flat, y_train)
        train_time = time.time() - start

        y_pred = pipeline.predict(X_val_flat)
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average="macro")

        mlflow.log_metric("val_accuracy", acc)
        mlflow.log_metric("val_macro_f1", f1)
        mlflow.log_metric("train_time_seconds", train_time)

        logger.info(f"Training time: {train_time:.1f}s")
        logger.info(f"Val Accuracy: {acc:.4f}")
        logger.info(f"Val Macro F1: {f1:.4f}")

        model_path = "models/mlp_model.pkl"
        joblib.dump(pipeline, model_path)
        mlflow.sklearn.log_model(pipeline, "mlp_model")

        metadata = {
            "model_type": "MLP",
            "feature_type": "flattened_pixels",
            "train_date": datetime.now().isoformat(),
            "val_accuracy": acc,
            "val_macro_f1": f1,
            "train_time_seconds": train_time,
            "class_names": class_names,
            "mlflow_run_id": mlflow.active_run().info.run_id,
            "params": p
        }
        with open("models/mlp_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        run_id = mlflow.active_run().info.run_id

    os.makedirs("metrics", exist_ok=True)
    metrics = {
        "val_accuracy": acc,
        "val_macro_f1": f1,
        "train_time_seconds": train_time,
        "mlflow_run_id": run_id
    }
    with open("metrics/mlp_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"MLP model saved to {model_path}")
    return metrics


# ─────────────────────────────────────────
# MODEL C: Custom CNN (PyTorch)
# ─────────────────────────────────────────
def train_cnn(config: dict, params: dict) -> dict:
    """Train a simple custom CNN with PyTorch."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from sklearn.metrics import f1_score, accuracy_score
    from PIL import Image

    logger.info("="*50)
    logger.info("Training Model C: Custom CNN (PyTorch)")
    logger.info("="*50)

    p = params["train_cnn"]
    prep = params["prepare"]
    v1_dir = Path(config["data"]["processed_dir"]) / "v1"
    image_size = prep["image_size"]
    max_per_class = prep["max_images_per_class"]

    torch.manual_seed(p["random_seed"])

    with open(v1_dir / "class_names.json") as f:
        class_map = json.load(f)
    class_names = list(class_map.keys())
    num_classes = len(class_names)

    # Dataset class
    # Assisted by Claude AI for PyTorch Dataset boilerplate
    class PlantDataset(Dataset):
        def __init__(self, split_dir, class_names,
                     max_per_class, transform=None):
            self.samples = []
            self.transform = transform
            import random
            random.seed(42)
            for label_idx, class_name in enumerate(class_names):
                class_dir = Path(split_dir) / class_name
                if not class_dir.exists():
                    continue
                images = list(class_dir.glob("*.jpg")) + \
                    list(class_dir.glob("*.JPG"))
                random.shuffle(images)
                images = images[:max_per_class]
                for img_path in images:
                    self.samples.append((img_path, label_idx))

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            img_path, label = self.samples[idx]
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, label

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])

    # DataLoaders
    train_ds = PlantDataset(
        v1_dir / "train", class_names,
        max_per_class, train_transform)
    val_ds = PlantDataset(
        v1_dir / "val", class_names,
        max_per_class, val_transform)

    train_loader = DataLoader(
        train_ds, batch_size=p["batch_size"], shuffle=True)
    val_loader = DataLoader(
        val_ds, batch_size=p["batch_size"], shuffle=False)

    logger.info(f"Train samples: {len(train_ds)}")
    logger.info(f"Val samples: {len(val_ds)}")

    # Custom CNN Model
    class PlantCNN(nn.Module):
        def __init__(self, num_classes, num_filters):
            super().__init__()
            self.features = nn.Sequential(
                # Block 1
                nn.Conv2d(3, num_filters[0], 3, padding=1),
                nn.BatchNorm2d(num_filters[0]),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                # Block 2
                nn.Conv2d(num_filters[0], num_filters[1], 3,
                          padding=1),
                nn.BatchNorm2d(num_filters[1]),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                # Block 3
                nn.Conv2d(num_filters[1], num_filters[2], 3,
                          padding=1),
                nn.BatchNorm2d(num_filters[2]),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
            )
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
                nn.Linear(num_filters[2] * 16, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x

    num_filters = p.get("num_filters", [32, 64, 128])
    model = PlantCNN(num_classes, num_filters)
    device = torch.device("cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=p["learning_rate"])
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=5, gamma=0.5)

    mlflow.set_experiment(
        config["deployment"]["mlflow_experiment_name"])

    parent_run_id = os.environ.get("MLFLOW_RUN_ID")
    with mlflow.start_run(run_name="Custom_CNN",
                          run_id=parent_run_id if parent_run_id else None,
                          nested=True if parent_run_id else False):
        mlflow.log_param("model_type", "CustomCNN")
        mlflow.log_param("architecture", "3-block CNN")
        mlflow.log_param("num_filters", str(num_filters))
        mlflow.log_param("learning_rate", p["learning_rate"])
        mlflow.log_param("epochs", p["epochs"])
        mlflow.log_param("batch_size", p["batch_size"])
        mlflow.log_param("max_per_class", max_per_class)
        mlflow.log_param("num_classes", num_classes)
        mlflow.log_param("train_samples", len(train_ds))

        start = time.time()
        best_f1 = 0.0
        best_model_state = None

        for epoch in range(p["epochs"]):
            # Training
            model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            scheduler.step()
            avg_train_loss = train_loss / len(train_loader)

            # Validation
            model.eval()
            all_preds, all_labels = [], []
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(device)
                    batch_y = batch_y.to(device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    preds = torch.argmax(outputs, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(batch_y.cpu().numpy())

            avg_val_loss = val_loss / len(val_loader)
            acc = accuracy_score(all_labels, all_preds)
            f1 = f1_score(
                all_labels, all_preds, average="macro")

            # Log per-epoch metrics
            mlflow.log_metric("train_loss", avg_train_loss,
                              step=epoch)
            mlflow.log_metric("val_loss", avg_val_loss,
                              step=epoch)
            mlflow.log_metric("val_accuracy", acc, step=epoch)
            mlflow.log_metric("val_macro_f1", f1, step=epoch)

            epoch_msg = (
                f"Epoch {epoch+1:02d}/{p['epochs']} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"Acc: {acc:.4f} | F1: {f1:.4f}"
            )
            print(epoch_msg, flush=True)
            logger.info(epoch_msg)

            # Save best model
            if f1 > best_f1:
                best_f1 = f1
                best_model_state = model.state_dict().copy()

        train_time = time.time() - start

        # Save best model
        import torch
        model.load_state_dict(best_model_state)
        model_path = "models/cnn_model.pt"
        torch.save({
            "model_state_dict": best_model_state,
            "class_names": class_names,
            "num_classes": num_classes,
            "num_filters": num_filters,
            "image_size": image_size,
            "params": p
        }, model_path)

        mlflow.log_metric("best_val_macro_f1", best_f1)
        mlflow.log_metric("train_time_seconds", train_time)
        mlflow.pytorch.log_model(model, "cnn_model")

        metadata = {
            "model_type": "CustomCNN",
            "architecture": "3-block CNN",
            "train_date": datetime.now().isoformat(),
            "best_val_macro_f1": best_f1,
            "train_time_seconds": train_time,
            "class_names": class_names,
            "mlflow_run_id": mlflow.active_run().info.run_id,
            "params": p
        }
        with open("models/cnn_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        run_id = mlflow.active_run().info.run_id

    os.makedirs("metrics", exist_ok=True)
    metrics = {
        "val_accuracy": acc,
        "val_macro_f1": best_f1,
        "train_time_seconds": train_time,
        "mlflow_run_id": run_id
    }
    with open("metrics/cnn_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Best Val Macro F1: {best_f1:.4f}")
    logger.info(f"CNN model saved to {model_path}")
    return metrics


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["svm", "mlp", "cnn"],
        required=True,
        help="Which model to train"
    )
    args = parser.parse_args()

    config = load_config()
    params = load_params()

    # Set MLflow tracking URI
    mlflow.set_tracking_uri(
        config["deployment"]["mlflow_tracking_uri"])

    logger.info(f"Training model: {args.model.upper()}")
    logger.info(f"MLflow URI: "
                f"{config['deployment']['mlflow_tracking_uri']}")

    if args.model == "svm":
        metrics = train_svm(config, params)
    elif args.model == "mlp":
        metrics = train_mlp(config, params)
    elif args.model == "cnn":
        metrics = train_cnn(config, params)

    logger.info(f"Final metrics: {metrics}")
    logger.info("Training complete.")
