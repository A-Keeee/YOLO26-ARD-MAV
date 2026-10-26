"""Prepare local ARD-MAV crops, or explicitly select the original video conversion configuration."""

import argparse
import csv
import random
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import yaml


def prepare_local(config, settings, data):
    """Crop the cleaned training/validation images and preserve source coordinates in a manifest."""
    from ultralytics.utils.ops import xywh2xyxy, xyxy2xywh

    source_data = yaml.safe_load((config.parent / settings["source_data"]).read_text(encoding="utf-8"))
    source = Path(source_data["path"]).expanduser().resolve()
    output = Path(data["path"]).expanduser().resolve()
    size, ratio = settings["crop_size"], settings["background_ratio"]
    if not isinstance(size, int) or size < 1 or not 0 <= ratio <= 1 or data["names"] != source_data["names"]:
        raise ValueError("Check crop_size, background_ratio and matching dataset classes")
    splits = yaml.safe_load((source / "splits.yaml").read_text(encoding="utf-8"))
    if set(splits["train"]) & set(splits["val"]):
        raise ValueError("Training and validation videos must be disjoint")
    inputs = {split: sorted((source / source_data[split]).glob("*.jpg")) for split in ("train", "val")}
    for split, paths in inputs.items():
        labels = list((source / "labels" / split).glob("*.txt"))
        if not paths or {p.stem for p in paths} != {p.stem for p in labels}:
            raise ValueError(f"Missing source image/label pairs: {split}")
    output.mkdir(parents=True, exist_ok=False)
    for split in inputs:
        (output / data[split]).mkdir(parents=True)
        (output / "labels" / split).mkdir(parents=True)
    cv2.setNumThreads(1)

    def crop_image(item):
        split, path = item
        image = cv2.imread(str(path))
        if image is None or min(image.shape[:2]) < size:
            raise ValueError(f"Cannot decode image or crop exceeds image size: {path}")
        height, width = image.shape[:2]
        label = source / "labels" / split / f"{path.stem}.txt"
        rows = np.array([list(map(float, row.split())) for row in label.read_text().splitlines() if row.strip()])
        rows = rows.reshape(-1, 5)
        boxes = xywh2xyxy(rows[:, 1:]) * np.array([width, height, width, height])
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
        rng = random.Random(f"{settings['seed']}:{split}:{path.stem}")
        windows = []
        for index, box in enumerate(boxes):
            x = min(max(round((box[0] + box[2] - size) / 2), 0), width - size)
            y = min(max(round((box[1] + box[3] - size) / 2), 0), height - size)
            windows.append((x, y, index))
        if not len(boxes) or rng.random() < ratio:
            for _ in range(50):
                x, y = rng.randint(0, width - size), rng.randint(0, height - size)
                overlap = (boxes[:, 0] < x + size) & (boxes[:, 2] > x)
                overlap &= (boxes[:, 1] < y + size) & (boxes[:, 3] > y)
                if not overlap.any():
                    windows.append((x, y, -1))
                    break
            else:
                raise ValueError(f"Cannot sample a background crop without annotated objects: {path}")
        records = []
        for x, y, anchor in windows:
            clipped = (boxes - np.array([x, y, x, y])).clip(0, size)
            keep = (clipped[:, 2] > clipped[:, 0]) & (clipped[:, 3] > clipped[:, 1])
            normalized = xyxy2xywh(clipped[keep]) / size
            text = "\n".join(
                f"{int(cls)} " + " ".join(f"{value:.8f}" for value in box)
                for cls, box in zip(rows[keep, 0], normalized)
            )
            name = f"{path.stem}_{'bg' if anchor < 0 else f'obj{anchor}'}"
            destination = output / data[split] / f"{name}.jpg"
            if not cv2.imwrite(str(destination), image[y : y + size, x : x + size]):
                raise OSError(f"Cannot write crop: {destination}")
            (output / "labels" / split / f"{name}.txt").write_text(text, encoding="utf-8")
            records.append(
                [split, str(path), str(destination.relative_to(output)), x, y, size, anchor, int(keep.sum())]
            )
        return records

    summary = {"source": str(source), "settings": settings, "source_videos": splits, "splits": {}}
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["split", "source_image", "crop_image", "x", "y", "size", "anchor", "objects"])
        for split, paths in inputs.items():
            stats = {"source_images": len(paths), "images": 0, "objects": 0, "background_images": 0}
            with ThreadPoolExecutor(max_workers=settings["workers"]) as pool:
                for count, records in enumerate(pool.map(crop_image, ((split, p) for p in paths)), 1):
                    writer.writerows(records)
                    stats["images"] += len(records)
                    stats["objects"] += sum(row[-1] for row in records)
                    stats["background_images"] += sum(row[-1] == 0 for row in records)
                    if count % 2000 == 0 or count == len(paths):
                        print(f"{split}: {count}/{len(paths)} source images, {stats['images']} crops", flush=True)
            summary["splits"][split] = stats
    summary["validation"] = "Ground-truth-centered crops for local training monitoring only, not end-to-end metrics."
    summary["test"] = "No cropped test set; use original full frames/videos for final evaluation."
    (output / "summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")
    print(yaml.safe_dump(summary["splits"], sort_keys=False), flush=True)


def main():
    """Prepare disjoint video splits while preserving the original videos and XML annotations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/prepare_local.yaml"
    )
    args = parser.parse_args()
    config = args.config.expanduser().resolve()
    settings = yaml.safe_load(config.read_text(encoding="utf-8"))
    data = yaml.safe_load((config.parent / settings["data"]).read_text(encoding="utf-8"))
    if settings.get("mode") == "local":
        prepare_local(config, settings, data)
        return
    source = Path(settings["source"]).expanduser().resolve()
    output = Path(data["path"]).expanduser().resolve()
    videos = sorted(p.name for p in (source / "Annotations").iterdir() if p.is_dir())
    test = settings["test_videos"]
    if not set(test).issubset(videos):
        raise ValueError("Some test videos have no annotation directory")
    pool = [video for video in videos if video not in test]
    random.Random(settings["seed"]).shuffle(pool)
    n_val, stride = settings["val_videos"], settings["frame_stride"]
    if not 0 < n_val < len(pool) or not isinstance(stride, int) or stride < 1:
        raise ValueError("val_videos must leave train videos, and frame_stride must be a positive integer")
    splits = {"train": pool[n_val:], "val": pool[:n_val], "test": test}
    class_ids = {name: index for index, name in data["names"].items()}
    output.mkdir(parents=True, exist_ok=False)  # Use a new output directory to avoid mixing different splits.

    for split, sequence_names in splits.items():
        image_dir, label_dir = output / data[split], output / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        for sequence in sequence_names:
            annotations = {
                int(p.stem.rsplit("_", 1)[1]): p
                for p in (source / "Annotations" / sequence).glob("*.xml")
                if (int(p.stem.rsplit("_", 1)[1]) - 1) % stride == 0
            }
            capture = cv2.VideoCapture(str(source / "videos" / f"{sequence}.mp4"))
            try:
                for frame_id in range(1, max(annotations) + 1):
                    if not capture.grab():
                        raise OSError(f"Cannot decode {sequence}, frame {frame_id}")
                    if frame_id not in annotations:
                        continue
                    ok, frame = capture.retrieve()
                    if not ok:
                        raise OSError(f"Cannot retrieve {sequence}, frame {frame_id}")
                    xml = annotations[frame_id]
                    annotation = ET.parse(xml).getroot()
                    width, height = (int(annotation.findtext(f"size/{key}")) for key in ("width", "height"))
                    if frame.shape[:2] != (height, width):
                        raise ValueError(f"Video and XML image sizes differ: {xml}")
                    labels = []
                    for obj in annotation.findall("object"):
                        xmin, ymin, xmax, ymax = (
                            float(obj.findtext(f"bndbox/{key}")) for key in ("xmin", "ymin", "xmax", "ymax")
                        )
                        # Match the VOC conversion used in ultralytics/cfg/datasets/VOC.yaml.
                        box = ((xmin + xmax) / 2 - 1, (ymin + ymax) / 2 - 1, xmax - xmin, ymax - ymin)
                        normalized = [value / size for value, size in zip(box, (width, height, width, height))]
                        labels.append(f"{class_ids[obj.findtext('name')]} " + " ".join(f"{x:.8f}" for x in normalized))
                    if not cv2.imwrite(str(image_dir / f"{xml.stem}.jpg"), frame):
                        raise OSError(f"Cannot write image for {xml}")
                    (label_dir / f"{xml.stem}.txt").write_text("\n".join(labels), encoding="utf-8")
            finally:
                capture.release()
            print(f"{split}: {sequence}, {len(annotations)} images", flush=True)
    (output / "splits.yaml").write_text(yaml.safe_dump(splits, sort_keys=False), encoding="utf-8")
    print(f"Dataset ready: {output}")


if __name__ == "__main__":
    main()
