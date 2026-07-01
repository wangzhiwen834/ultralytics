# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Train YOLO11n-LSVD on the DAIR-V2X YOLO dataset.

Run from the repository root:
    python train_lsvd.py

The script uses the current Python interpreter, so it works with the conda environment selected in the editor.
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from pathlib import Path

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
RAW_DATASET = WORKSPACE / "dataset"
PREPARED_DATASET = WORKSPACE / "dataset_lsvd"
DEFAULT_MODEL = ROOT / "ultralytics" / "cfg" / "models" / "11" / "yolo11n_lsvd.yaml"
DEFAULT_DATA = WORKSPACE / "dataset" / "DAIRV2X_YOLO_NO_TRAFFICCONE" / "data.yaml"
DEFAULT_WORKERS = min(4, os.cpu_count() or 1)
CLASS_NAMES = [
    "Car",
    "Pedestrian",
    "Cyclist",
    "Motorcyclist",
    "Bus",
    "Truck",
    "Van",
    "Barrowlist",
]
IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    """Parse training arguments."""
    parser = argparse.ArgumentParser(description="Train YOLO11n-LSVD.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Path to YOLO11n-LSVD YAML.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to dataset YAML.")
    parser.add_argument("--source", type=Path, default=RAW_DATASET / "test", help="Legacy Roboflow split containing images/labels.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size.")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Dataloader workers. Use 0 if Windows multiprocessing is unstable.")
    parser.add_argument("--device", default=None, help="Training device, e.g. 0 or cpu. Defaults to CUDA if available else CPU.")
    parser.add_argument("--project", default=str(ROOT / "runs" / "lsvd"), help="Training output project directory.")
    parser.add_argument("--name", default="yolo11n_lsvd_dairv2x_no_trafficcone", help="Training run name.")
    parser.add_argument("--seed", type=int, default=42, help="Dataset split seed.")
    parser.add_argument("--iou-loss", default="wiou_v3", choices=["ciou", "wiou_v3"], help="BBox regression loss.")
    parser.add_argument("--cache", default="false", choices=["false", "ram", "disk"], help="Image cache mode. Use disk/ram only when resources allow.")
    parser.add_argument("--prepare", action="store_true", help="Prepare the legacy Roboflow split before training.")
    parser.add_argument("--no-prepare", action="store_true", help="Deprecated alias kept for old commands.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow overwriting existing run directory.")
    return parser.parse_args()


def image_label_pairs(source: Path) -> list[tuple[Path, Path]]:
    image_dir = source / "images"
    label_dir = source / "labels"
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")

    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMG_SUFFIXES)
    pairs = [(p, label_dir / f"{p.stem}.txt") for p in images if (label_dir / f"{p.stem}.txt").exists()]
    if not pairs:
        raise FileNotFoundError(f"No image/label pairs found under: {source}")
    missing = len(images) - len(pairs)
    if missing:
        print(f"WARNING: skipped {missing} images without matching labels.")
    return pairs


def copy_split(items: list[tuple[Path, Path]], split_dir: Path) -> None:
    """Copy image/label pairs into a YOLO split directory."""
    image_dir = split_dir / "images"
    label_dir = split_dir / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    for image, label in items:
        shutil.copy2(image, image_dir / image.name)
        shutil.copy2(label, label_dir / label.name)


def prepare_dataset(source: Path, output: Path, seed: int) -> Path:
    """Create train/valid/test splits from a single Roboflow split without modifying the raw dataset."""
    pairs = image_label_pairs(source)
    random.Random(seed).shuffle(pairs)
    n = len(pairs)
    n_train = max(1, int(n * 0.8))
    n_val = max(1, int(n * 0.1))
    splits = {
        "train": pairs[:n_train],
        "valid": pairs[n_train : n_train + n_val],
        "test": pairs[n_train + n_val :],
    }

    for split, items in splits.items():
        copy_split(items, output / split)

    data_yaml = output / "data.yaml"
    names = ", ".join(CLASS_NAMES)
    data_yaml.write_text(
        f"path: {output.as_posix()}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: [{names}]\n",
        encoding="utf-8",
    )
    print(f"Prepared dataset: {output}")
    print({split: len(items) for split, items in splits.items()})
    return data_yaml


def validate_paths(model: Path, data: Path) -> None:
    """Validate required files before training."""
    if not model.exists():
        raise FileNotFoundError(f"Model YAML not found: {model}")
    if not data.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data}")


def main() -> None:
    """Train YOLO11n-LSVD."""
    args = parse_args()
    device = args.device
    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"

    if args.prepare and not args.no_prepare:
        # DAIR-V2X is already in YOLO format. Preparation is only for the old single-split Roboflow workflow.
        output = args.data.parent if args.data != DEFAULT_DATA else PREPARED_DATASET
        args.data = prepare_dataset(args.source, output, args.seed)

    validate_paths(args.model, args.data)

    print(f"Python: {sys.executable}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Data: {args.data}")

    model = YOLO(str(args.model))
    cache = False if args.cache == "false" else args.cache
    results = model.train(
        data=str(args.data),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        device=device,
        iou_loss=args.iou_loss,
        cache=cache,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
    )
    print(results)


if __name__ == "__main__":
    main()
