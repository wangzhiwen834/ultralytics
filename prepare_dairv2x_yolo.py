# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Convert DAIR-V2X-I infrastructure-side labels to YOLO detection format."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


CLASSES = ["Car", "Pedestrian", "Cyclist", "Motorcyclist", "Bus", "Truck", "Van", "Barrowlist"]
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    root = Path(__file__).resolve().parents[1] / "dataset" / "DAIRV2X"
    parser = argparse.ArgumentParser(description="Prepare DAIR-V2X-I as a YOLO-format dataset.")
    parser.add_argument("--source", type=Path, default=root, help="Path to the extracted DAIRV2X directory.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "dataset" / "DAIRV2X_YOLO_NO_TRAFFICCONE",
        help="Output YOLO dataset directory.",
    )
    parser.add_argument("--copy-images", action="store_true", help="Copy images instead of creating hardlinks.")
    return parser.parse_args()


def dair_paths(source: Path) -> dict[str, Path]:
    """Return the important DAIR-V2X-I paths from the extracted package layout."""
    full = source / "DAIR-V2X-I" / "Full Dataset (train&val)" / "single-infrastructure-side" / "single-infrastructure-side"
    return {
        "full": full,
        "images": source
        / "DAIR-V2X-I"
        / "Full Dataset (train&val)"
        / "single-infrastructure-side-image"
        / "single-infrastructure-side-image",
        "labels": full / "label" / "camera",
        "calib": full / "calib" / "camera_intrinsic",
        "split": full / "split_data.json",
    }


def clamp(value: float, lower: float, upper: float) -> float:
    """Clamp a coordinate to image bounds."""
    return max(lower, min(value, upper))


def image_size(calib_file: Path) -> tuple[int, int]:
    """Read image size from per-frame camera calibration, with DAIR defaults as fallback."""
    if not calib_file.exists():
        return IMAGE_WIDTH, IMAGE_HEIGHT
    calib = json.loads(calib_file.read_text(encoding="utf-8"))
    return int(calib.get("width", IMAGE_WIDTH)), int(calib.get("height", IMAGE_HEIGHT))


def convert_label(label_file: Path, calib_file: Path, class_to_id: dict[str, int]) -> list[str]:
    """Convert one DAIR camera-label JSON file to YOLO normalized xywh rows."""
    width, height = image_size(calib_file)
    objects = json.loads(label_file.read_text(encoding="utf-8"))
    rows = []
    for obj in objects:
        cls = obj.get("type")
        if cls not in class_to_id:
            continue

        box = obj.get("2d_box", {})
        xmin = clamp(float(box["xmin"]), 0.0, float(width))
        ymin = clamp(float(box["ymin"]), 0.0, float(height))
        xmax = clamp(float(box["xmax"]), 0.0, float(width))
        ymax = clamp(float(box["ymax"]), 0.0, float(height))
        bw = xmax - xmin
        bh = ymax - ymin
        if bw <= 0.0 or bh <= 0.0:
            continue

        cx = (xmin + xmax) / 2.0 / width
        cy = (ymin + ymax) / 2.0 / height
        nw = bw / width
        nh = bh / height
        rows.append(f"{class_to_id[cls]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return rows


def place_image(src: Path, dst: Path, copy_images: bool) -> None:
    """Place an image in the YOLO dataset using a hardlink by default."""
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy_images:
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except FileExistsError:
        return
    except OSError:
        shutil.copy2(src, dst)


def write_yaml(output: Path) -> None:
    """Write Ultralytics-compatible data.yaml."""
    lines = [
        f"path: {output.as_posix()}",
        "train: train/images",
        "val: valid/images",
        "",
        f"nc: {len(CLASSES)}",
        "names:",
    ]
    lines.extend(f"  {i}: {name}" for i, name in enumerate(CLASSES))
    (output / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_split(split_name: str, ids: list[str], paths: dict[str, Path], output: Path, copy_images: bool) -> tuple[int, int]:
    """Prepare one YOLO split and return image/label counts."""
    out_split = "valid" if split_name == "val" else split_name
    image_dir = output / out_split / "images"
    label_dir = output / out_split / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    class_to_id = {name: i for i, name in enumerate(CLASSES)}
    image_count = 0
    label_count = 0
    for frame_id in ids:
        image_file = paths["images"] / f"{frame_id}.jpg"
        label_file = paths["labels"] / f"{frame_id}.json"
        if not image_file.exists() or not label_file.exists():
            continue

        place_image(image_file, image_dir / image_file.name, copy_images)
        rows = convert_label(label_file, paths["calib"] / f"{frame_id}.json", class_to_id)
        (label_dir / f"{frame_id}.txt").write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
        image_count += 1
        label_count += 1
    return image_count, label_count


def main() -> None:
    """Convert the DAIR-V2X train/val annotations into a directly trainable YOLO dataset."""
    args = parse_args()
    source = Path(args.source)
    output = Path(args.output)
    paths = dair_paths(source)
    split = json.loads(paths["split"].read_text(encoding="utf-8"))

    summary = {}
    for split_name in ("train", "val"):
        images, labels = prepare_split(split_name, split[split_name], paths, output, bool(args.copy_images))
        summary[split_name] = {"images": images, "labels": labels}
    write_yaml(output)

    print(f"YOLO dataset saved to: {output}")
    print(f"Classes ({len(CLASSES)}): {', '.join(CLASSES)}")
    print(f"Summary: {summary}")
    print(f"Data YAML: {output / 'data.yaml'}")


if __name__ == "__main__":
    main()
