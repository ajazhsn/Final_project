# src/transform.py
import os
import json
import yaml
import random
import logging
import shutil
from pathlib import Path
from datetime import datetime
# Assisted by Claude AI for augmentation boilerplate
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[
        logging.FileHandler("logs/transform.log"),
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


def augment_image(img: Image.Image, params: dict) -> list:
    """
    Generate augmented versions of a single image.
    Returns list of augmented PIL images.
    """
    augmented = []

    # Original
    augmented.append(img)

    # Horizontal flip
    if params["transform"]["horizontal_flip"]:
        augmented.append(img.transpose(Image.FLIP_LEFT_RIGHT))

    # Rotation
    degrees = params["transform"]["rotation_degrees"]
    for angle in [-degrees, degrees]:
        rotated = img.rotate(angle, fillcolor=(0, 0, 0))
        augmented.append(rotated)

    # Color jitter — brightness
    jitter = params["transform"]["color_jitter"]
    enhancer = ImageEnhance.Brightness(img)
    augmented.append(enhancer.enhance(1 + jitter))
    augmented.append(enhancer.enhance(1 - jitter))

    # Color jitter — contrast
    enhancer = ImageEnhance.Contrast(img)
    augmented.append(enhancer.enhance(1 + jitter))

    return augmented


def transform_dataset(config: dict, params: dict) -> dict:
    """
    Creates v2_augmented from v1 processed data.
    Only augments the training set — val and test remain unchanged.
    """
    processed_dir = Path(config["data"]["processed_dir"])
    v1_dir = processed_dir / "v1"
    v2_dir = processed_dir / "v2"

    if not v1_dir.exists():
        raise FileNotFoundError(
            f"v1 processed data not found at {v1_dir}. Run data_prep.py first.")

    # Load class names from v1
    with open(v1_dir / "class_names.json", "r") as f:
        class_map = json.load(f)

    class_names = list(class_map.keys())
    logger.info(f"Classes found: {class_names}")

    # Create v2 directory structure
    for split in ["train", "val", "test"]:
        for class_name in class_names:
            (v2_dir / split / class_name).mkdir(parents=True, exist_ok=True)

    stats = {
        "version": "v2",
        "augmentation": params["transform"],
        "classes": [],
        "total": {"train": 0, "val": 0, "test": 0},
        "created_at": datetime.now().isoformat()
    }

    # Copy val and test unchanged
    for split in ["val", "test"]:
        logger.info(f"Copying {split} set unchanged...")
        for class_name in class_names:
            src_dir = v1_dir / split / class_name
            dst_dir = v2_dir / split / class_name
            if not src_dir.exists():
                continue
            images = list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.JPG"))
            for img_path in images:
                shutil.copy2(img_path, dst_dir / img_path.name)
                stats["total"][split] += 1

    # Augment training set only
    logger.info("Augmenting training set...")
    random.seed(params["prepare"]["random_seed"])

    for class_name in class_names:
        src_dir = v1_dir / "train" / class_name
        dst_dir = v2_dir / "train" / class_name

        if not src_dir.exists():
            logger.warning(f"Train dir not found for {class_name}, skipping")
            continue

        images = list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.JPG"))
        class_count = 0

        for img_path in tqdm(images, desc=f"Augmenting {class_name}"):
            try:
                with Image.open(img_path) as img:
                    img = img.convert("RGB")
                    augmented_versions = augment_image(img, params)

                    for idx, aug_img in enumerate(augmented_versions):
                        stem = img_path.stem
                        dest = dst_dir / f"{stem}_aug{idx}.jpg"
                        aug_img.save(dest, "JPEG", quality=95)
                        class_count += 1
            except Exception as e:
                logger.error(f"Failed to augment {img_path}: {e}")

        stats["total"]["train"] += class_count
        stats["classes"].append({
            "name": class_name,
            "train_augmented": class_count
        })
        logger.info(f"  {class_name}: {len(images)} → {class_count} images")

    # Copy class names to v2
    shutil.copy2(v1_dir / "class_names.json", v2_dir / "class_names.json")

    # Save stats
    with open(v2_dir / "split_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"\n{'='*50}")
    logger.info(f"Augmentation complete!")
    logger.info(
        f"Train: {stats['total']['train']} | Val: {stats['total']['val']} | Test: {stats['total']['test']}")
    logger.info(f"Output: {v2_dir}")
    logger.info(f"{'='*50}")

    return stats


def update_manifest(stats: dict) -> None:
    """Update manifest with augmentation details."""
    entry = (
        f"v2_augmented | {datetime.now().strftime('%Y-%m-%d')} | "
        f"src/transform.py | data/processed/v1/train/ | "
        f"data/processed/v2/ | "
        f"Augmented train set — Train:{stats['total']['train']} "
        f"Val:{stats['total']['val']} Test:{stats['total']['test']}\n"
    )
    with open("manifest.txt", "a") as f:
        f.write(entry)
    logger.info("Manifest updated.")


if __name__ == "__main__":
    logger.info("Starting augmentation pipeline...")
    config = load_config()
    params = load_params()
    stats = transform_dataset(config, params)
    update_manifest(stats)
    logger.info("Done.")
