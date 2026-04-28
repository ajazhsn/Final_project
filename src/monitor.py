# src/monitor.py
# Data Drift Monitoring for Plant Disease Detection
import os
import json
import yaml
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[
        logging.FileHandler("logs/monitor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def extract_image_stats(image_path: str, image_size: int = 224) -> dict:
    """Extract statistical features from a single image."""
    with Image.open(image_path) as img:
        img = img.convert("RGB").resize((image_size, image_size))
        arr = np.array(img) / 255.0
        return {
            "mean_r": float(arr[:, :, 0].mean()),
            "mean_g": float(arr[:, :, 1].mean()),
            "mean_b": float(arr[:, :, 2].mean()),
            "std_r": float(arr[:, :, 0].std()),
            "std_g": float(arr[:, :, 1].std()),
            "std_b": float(arr[:, :, 2].std()),
            "brightness": float(arr.mean()),
            "contrast": float(arr.std()),
        }


def compute_baseline_stats(data_dir: str,
                           image_size: int = 224,
                           max_per_class: int = 50) -> dict:
    """Compute baseline statistics from training data."""
    logger.info("Computing baseline statistics from training data...")
    data_dir = Path(data_dir)
    all_stats = []

    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        images = list(class_dir.glob("*.jpg")) + \
                 list(class_dir.glob("*.JPG"))
        images = images[:max_per_class]

        for img_path in images:
            try:
                stats = extract_image_stats(
                    str(img_path), image_size)
                all_stats.append(stats)
            except Exception as e:
                logger.error(f"Failed: {img_path}: {e}")

    if not all_stats:
        raise ValueError("No images found for baseline")

    # Aggregate statistics
    baseline = {}
    for key in all_stats[0].keys():
        values = [s[key] for s in all_stats]
        baseline[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values))
        }

    baseline["num_images"] = len(all_stats)
    baseline["computed_at"] = datetime.now().isoformat()

    logger.info(f"Baseline computed from {len(all_stats)} images")
    return baseline


def compute_production_stats(prod_dir: str,
                             image_size: int = 224) -> dict:
    """Compute statistics from production/new data."""
    logger.info("Computing production data statistics...")
    prod_dir = Path(prod_dir)
    all_stats = []

    images = (list(prod_dir.glob("**/*.jpg")) +
              list(prod_dir.glob("**/*.JPG")) +
              list(prod_dir.glob("**/*.png")))

    if not images:
        logger.warning(f"No images found in {prod_dir}")
        return {}

    for img_path in images:
        try:
            stats = extract_image_stats(str(img_path), image_size)
            all_stats.append(stats)
        except Exception as e:
            logger.error(f"Failed: {img_path}: {e}")

    prod_stats = {}
    for key in all_stats[0].keys():
        values = [s[key] for s in all_stats]
        prod_stats[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values))
        }

    prod_stats["num_images"] = len(all_stats)
    prod_stats["computed_at"] = datetime.now().isoformat()

    logger.info(
        f"Production stats computed from {len(all_stats)} images")
    return prod_stats


def detect_drift(baseline: dict, production: dict,
                 threshold: float = 0.15) -> dict:
    """
    Detect drift by comparing means.
    Drift is flagged if the relative difference exceeds threshold.
    """
    logger.info("Detecting data drift...")
    drift_report = {
        "timestamp": datetime.now().isoformat(),
        "threshold": threshold,
        "features": {},
        "drift_detected": False,
        "drifted_features": []
    }

    feature_keys = [
        "mean_r", "mean_g", "mean_b",
        "std_r", "std_g", "std_b",
        "brightness", "contrast"
    ]

    for key in feature_keys:
        if key not in baseline or key not in production:
            continue

        baseline_mean = baseline[key]["mean"]
        prod_mean = production[key]["mean"]

        if baseline_mean == 0:
            relative_diff = abs(prod_mean)
        else:
            relative_diff = abs(
                prod_mean - baseline_mean) / abs(baseline_mean)

        is_drifted = relative_diff > threshold

        drift_report["features"][key] = {
            "baseline_mean": round(baseline_mean, 4),
            "production_mean": round(prod_mean, 4),
            "relative_difference": round(relative_diff, 4),
            "drifted": is_drifted
        }

        if is_drifted:
            drift_report["drifted_features"].append(key)
            drift_report["drift_detected"] = True
            logger.warning(
                f"DRIFT DETECTED in {key}: "
                f"baseline={baseline_mean:.4f} "
                f"production={prod_mean:.4f} "
                f"diff={relative_diff:.2%}")
        else:
            logger.info(
                f"No drift in {key}: "
                f"diff={relative_diff:.2%}")

    return drift_report


def simulate_production_data(train_dir: str,
                             prod_dir: str,
                             n_images: int = 20):
    """
    Simulate production data by copying and slightly
    modifying training images.
    """
    import shutil
    import random
    from PIL import ImageEnhance

    prod_path = Path(prod_dir)
    prod_path.mkdir(parents=True, exist_ok=True)

    train_path = Path(train_dir)
    all_images = []
    for class_dir in train_path.iterdir():
        if class_dir.is_dir():
            images = list(class_dir.glob("*.jpg"))
            all_images.extend(images)

    random.seed(42)
    selected = random.sample(
        all_images, min(n_images, len(all_images)))

    for i, img_path in enumerate(selected):
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            # Simulate slight brightness shift (drift)
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(
                random.uniform(0.7, 0.9))
            dest = prod_path / f"prod_{i:03d}.jpg"
            img.save(dest, "JPEG")

    logger.info(
        f"Simulated {len(selected)} production images in {prod_dir}")


def run_drift_monitoring():
    """Main drift monitoring function."""
    config = load_config()
    image_size = 224
    threshold = 0.15

    train_dir = "data/processed/v1/train"
    prod_dir = config["data"]["production_dir"]
    baseline_path = "data/baseline_stats.json"
    report_path = "metrics/drift_report.json"

    os.makedirs("metrics", exist_ok=True)
    os.makedirs(prod_dir, exist_ok=True)

    # Step 1 — Compute or load baseline
    if Path(baseline_path).exists():
        logger.info("Loading existing baseline stats...")
        with open(baseline_path) as f:
            baseline = json.load(f)
    else:
        logger.info("Computing baseline stats...")
        baseline = compute_baseline_stats(
            train_dir, image_size)
        with open(baseline_path, "w") as f:
            json.dump(baseline, f, indent=2)
        logger.info(f"Baseline saved to {baseline_path}")

    # Step 2 — Simulate production data if empty
    prod_images = list(Path(prod_dir).glob("**/*.jpg"))
    if not prod_images:
        logger.info("No production data found — simulating...")
        simulate_production_data(
            train_dir, prod_dir, n_images=30)

    # Step 3 — Compute production stats
    production = compute_production_stats(prod_dir, image_size)

    if not production:
        logger.error("Could not compute production stats")
        return

    # Step 4 — Detect drift
    drift_report = detect_drift(baseline, production, threshold)

    # Step 5 — Save report
    with open(report_path, "w") as f:
        json.dump(drift_report, f, indent=2)

    # Step 6 — Print summary
    logger.info(f"\n{'='*50}")
    logger.info("DRIFT MONITORING REPORT")
    logger.info(f"{'='*50}")
    logger.info(
        f"Baseline images: {baseline.get('num_images', 'N/A')}")
    logger.info(
        f"Production images: {production.get('num_images', 'N/A')}")
    logger.info(
        f"Drift detected: {drift_report['drift_detected']}")

    if drift_report["drifted_features"]:
        logger.warning(
            f"Drifted features: "
            f"{drift_report['drifted_features']}")
        logger.warning(
            "ACTION REQUIRED: Consider retraining the model!")
    else:
        logger.info(
            "No significant drift detected. Model is stable.")

    logger.info(f"Report saved to {report_path}")
    return drift_report


if __name__ == "__main__":
    logger.info("Starting data drift monitoring...")
    report = run_drift_monitoring()
    logger.info("Monitoring complete.")