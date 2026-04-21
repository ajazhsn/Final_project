# src/finetune.py
import os
import sys
import json
import yaml
import time
import argparse
import logging
import numpy as np
from pathlib import Path
from datetime import datetime

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[
        logging.FileHandler("logs/finetune.log"),
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


def finetune(args):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from sklearn.metrics import f1_score, accuracy_score
    from PIL import Image

    config = load_config()
    params = load_params()

    mlflow.set_tracking_uri(
        config["deployment"]["mlflow_tracking_uri"])

    v1_dir = Path(config["data"]["processed_dir"]) / "v1"
    image_size = params["prepare"]["image_size"]

    with open(v1_dir / "class_names.json") as f:
        class_map = json.load(f)
    class_names = list(class_map.keys())
    num_classes = len(class_names)

    logger.info(f"Loading registered model: "
                f"{args.registered_model}@{args.model_alias}")

    # Load registered model
    model_uri = (f"models:/{args.registered_model}"
                 f"@{args.model_alias}")
    loaded = mlflow.pytorch.load_model(model_uri)
    logger.info("Registered model loaded successfully!")

    # Dataset
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
                images = (list(class_dir.glob("*.jpg")) +
                          list(class_dir.glob("*.JPG")))
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

    train_ds = PlantDataset(
        v1_dir / "train", class_names,
        args.max_per_class, train_transform)
    val_ds = PlantDataset(
        v1_dir / "val", class_names,
        args.max_per_class, val_transform)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False)

    logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    device = torch.device("cpu")
    model = loaded.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=8, gamma=0.5)

    mlflow.set_experiment(
        config["deployment"]["mlflow_experiment_name"])

    parent_run_id = os.environ.get("MLFLOW_RUN_ID")
    with mlflow.start_run(
            run_name="CNN_Finetuned",
            run_id=parent_run_id if parent_run_id else None,
            nested=True if parent_run_id else False):

        mlflow.log_param("model_type", "CustomCNN_Finetuned")
        mlflow.log_param("base_model",
                         f"{args.registered_model}"
                         f"@{args.model_alias}")
        mlflow.log_param("learning_rate", args.learning_rate)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("max_per_class", args.max_per_class)
        mlflow.log_param("num_classes", num_classes)
        mlflow.log_param("train_samples", len(train_ds))

        start = time.time()
        best_f1 = 0.0
        best_model_state = None

        for epoch in range(args.epochs):
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
            f1 = f1_score(all_labels, all_preds,
                          average="macro")

            mlflow.log_metric("train_loss", avg_train_loss,
                              step=epoch)
            mlflow.log_metric("val_loss", avg_val_loss,
                              step=epoch)
            mlflow.log_metric("val_accuracy", acc, step=epoch)
            mlflow.log_metric("val_macro_f1", f1, step=epoch)

            epoch_msg = (
                f"Epoch {epoch+1:02d}/{args.epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"Acc: {acc:.4f} | F1: {f1:.4f}"
            )
            print(epoch_msg, flush=True)
            logger.info(epoch_msg)

            if f1 > best_f1:
                best_f1 = f1
                best_model_state = model.state_dict().copy()

        train_time = time.time() - start

        # Save best model
        model.load_state_dict(best_model_state)
        model_path = "models/cnn_finetuned.pt"

        # Load checkpoint info from original model
        checkpoint = torch.load("models/cnn_model.pt",
                                weights_only=True)
        torch.save({
            "model_state_dict": best_model_state,
            "class_names": class_names,
            "num_classes": num_classes,
            "num_filters": checkpoint.get(
                "num_filters", [32, 64, 128]),
            "image_size": image_size,
            "finetuned_from": (f"{args.registered_model}"
                               f"@{args.model_alias}"),
            "params": vars(args)
        }, model_path)

        mlflow.log_metric("best_val_macro_f1", best_f1)
        mlflow.log_metric("train_time_seconds", train_time)
        mlflow.pytorch.log_model(model, "cnn_finetuned")

        run_id = mlflow.active_run().info.run_id

        # Register new version if better
        client = MlflowClient()
        result = mlflow.register_model(
            f"runs:/{run_id}/cnn_finetuned",
            "PlantDiseaseDetector"
        )
        new_version = result.version

        if best_f1 > 0.868:
            client.set_registered_model_alias(
                "PlantDiseaseDetector",
                "champion",
                str(new_version)
            )
            logger.info(
                f"New champion! Version {new_version} "
                f"with F1: {best_f1:.4f}")
        else:
            logger.info(
                f"Version {new_version} registered "
                f"but champion unchanged (F1: {best_f1:.4f}"
                f" vs 0.868)")

        logger.info(f"Best Val Macro F1: {best_f1:.4f}")
        logger.info(f"Model saved: {model_path}")
        logger.info(f"Train time: {train_time:.1f}s")

    # Save metrics
    os.makedirs("metrics", exist_ok=True)
    metrics = {
        "val_accuracy": acc,
        "val_macro_f1": best_f1,
        "train_time_seconds": train_time,
        "mlflow_run_id": run_id,
        "finetuned_from": (f"{args.registered_model}"
                           f"@{args.model_alias}")
    }
    with open("metrics/cnn_finetuned_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float,
                        default=0.0005)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_per_class", type=int, default=150)
    parser.add_argument("--registered_model", type=str,
                        default="PlantDiseaseDetector")
    parser.add_argument("--model_alias", type=str,
                        default="champion")
    args = parser.parse_args()

    logger.info("Starting finetuning from registered model...")
    logger.info(f"Base model: {args.registered_model}"
                f"@{args.model_alias}")
    logger.info(f"LR: {args.learning_rate} | "
                f"Epochs: {args.epochs} | "
                f"Max per class: {args.max_per_class}")

    metrics = finetune(args)
    logger.info(f"Final metrics: {metrics}")
    logger.info("Finetuning complete.")
