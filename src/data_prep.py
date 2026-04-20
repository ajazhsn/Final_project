# src/data_prep.py
import os
import shutil
import yaml
import json
import random
import logging
from pathlib import Path
from datetime import datetime
# Assisted by Claude AI for boilerplate structure
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[
        logging.FileHandler("logs/data_prep.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from yaml file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_params(params_path: str = "params.yaml") -> dict:
    """Load DVC params from yaml file."""
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def validate_dataset(raw_path: str, min_images: int) -> list:
    """
    Validate dataset structure.
    Returns list of valid class folders with enough images.
    """
    raw_path = Path(raw_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data path not found: {raw_path}")

    valid_classes = []
    all_classes = [d for d in raw_path.iterdir() if d.is_dir()]

    logger.info(f"Found {len(all_classes)} class folders in {raw_path}")

    for class_dir in sorted(all_classes):
        images = list(class_dir.glob("*.jpg")) + \
            list(class_dir.glob("*.JPG")) + \
            list(class_dir.glob("*.png")) + \
            list(class_dir.glob("*.PNG"))

        if len(images) >= min_images:
            valid_classes.append((class_dir.name, images))
            logger.info(f"  ✓ {class_dir.name}: {len(images)} images")
        else:
            logger.warning(
                f"  ✗ {class_dir.name}: only {len(images)} images — skipping")

    logger.info(f"Valid classes: {len(valid_classes)}")
    return valid_classes


def resize_and_copy(
    image_path: Path,
    dest_path: Path,
    image_size: int
) -> bool:
    """Resize a single image and save to destination."""
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img = img.resize((image_size, image_size), Image.LANCZOS)
            img.save(dest_path, "JPEG", quality=95)
        return True
    except Exception as e:
        logger.error(f"Failed to process {image_path}: {e}")
        return False


def prepare_dataset(config: dict, params: dict) -> dict:
    """
    Main preparation function:
    - Reads raw data
    - Resizes images to target size
    - Splits into train/val/test
    - Saves to processed directory
    """
    raw_path = config["data"]["raw_path"]
    processed_dir = config["data"]["processed_dir"]
    version = config["data"]["current_version"]
    image_size = params["prepare"]["image_size"]
    test_split = params["prepare"]["test_split"]
    val_split = params["prepare"]["val_split"]
    random_seed = params["prepare"]["random_seed"]
    min_images = params["prepare"]["min_images_per_class"]

    random.seed(random_seed)

    # Output directories
    version_dir = Path(processed_dir) / version
    train_dir = version_dir / "train"
    val_dir = version_dir / "val"
    test_dir = version_dir / "test"

    for d in [train_dir, val_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Validate and get classes
    valid_classes = validate_dataset(raw_path, min_images)

    if not valid_classes:
        raise ValueError("No valid classes found in dataset!")

    # Track stats
    stats = {
        "version": version,
        "image_size": image_size,
        "random_seed": random_seed,
        "classes": [],
        "total": {"train": 0, "val": 0, "test": 0},
        "created_at": datetime.now().isoformat()
    }

    # Process each class
    for class_name, images in valid_classes:
        logger.info(f"Processing class: {class_name}")

        # Shuffle images
        images = list(images)
        random.shuffle(images)

        # Split
        n_total = len(images)
        n_test = max(1, int(n_total * test_split))
        n_val = max(1, int(n_total * val_split))
        n_train = n_total - n_test - n_val

        test_images = images[:n_test]
        val_images = images[n_test:n_test + n_val]
        train_images = images[n_test + n_val:]

        # Create class subdirectories
        for split_dir in [train_dir, val_dir, test_dir]:
            (split_dir / class_name).mkdir(parents=True, exist_ok=True)

        # Process each split
        split_counts = {"train": 0, "val": 0, "test": 0}

        for split_name, split_images, split_dir in [
            ("train", train_images, train_dir),
            ("val", val_images, val_dir),
            ("test", test_images, test_dir)
        ]:
            for img_path in tqdm(split_images, desc=f"{class_name}/{split_name}", leave=False):
                dest = split_dir / class_name / img_path.name
                if resize_and_copy(img_path, dest, image_size):
                    split_counts[split_name] += 1

        stats["classes"].append({
            "name": class_name,
            "train": split_counts["train"],
            "val": split_counts["val"],
            "test": split_counts["test"],
            "total": sum(split_counts.values())
        })

        for split in ["train", "val", "test"]:
            stats["total"][split] += split_counts[split]

        logger.info(
            f"  Train: {split_counts['train']} | Val: {split_counts['val']} | Test: {split_counts['test']}")

    # Save class names
    class_names = [c[0] for c in valid_classes]
    class_map = {name: idx for idx, name in enumerate(class_names)}

    with open(version_dir / "class_names.json", "w") as f:
        json.dump(class_map, f, indent=2)

    # Save stats
    with open(version_dir / "split_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"\n{'='*50}")
    logger.info(f"Dataset preparation complete!")
    logger.info(f"Version: {version}")
    logger.info(
        f"Train: {stats['total']['train']} | Val: {stats['total']['val']} | Test: {stats['total']['test']}")
    logger.info(f"Classes: {len(valid_classes)}")
    logger.info(f"Output: {version_dir}")
    logger.info(f"{'='*50}")

    return stats


def update_manifest(stats: dict, config: dict) -> None:
    """Update manifest.txt with processing details."""
    version = config["data"]["current_version"]
    total = sum(stats["total"].values())
    entry = (
        f"v1_resized | {datetime.now().strftime('%Y-%m-%d')} | "
        f"src/data_prep.py | data/raw/ | "
        f"data/processed/{version}/ | "
        f"Resized to {stats['image_size']}px — "
        f"Train:{stats['total']['train']} Val:{stats['total']['val']} Test:{stats['total']['test']}\n"
    )
    with open("manifest.txt", "a") as f:
        f.write(entry)
    logger.info("Manifest updated.")


if __name__ == "__main__":
    logger.info("Starting data preparation pipeline...")
    config = load_config()
    params = load_params()
    stats = prepare_dataset(config, params)
    update_manifest(stats, config)
    logger.info("Done.")
