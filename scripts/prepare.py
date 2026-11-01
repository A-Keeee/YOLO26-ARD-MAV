"""Prepare local ARD-MAV crops, or explicitly select the original video conversion configuration."""

import argparse
import csv
import math
import random
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path

import cv2
import numpy as np
import yaml


def multiscale_windows(boxes, shape, settings, split, rng):
    """Place complete targets at requested occupancies, rejecting crops that truncate another annotation."""
    height, width = shape[:2]
    windows, skipped = [], []
    for anchor, box in enumerate(boxes):
        long_side = float(max(box[2:] - box[:2]))
        if long_side + 1e-4 < settings["min_source_long_side"]:
            skipped.append((anchor, "", "below_min_source_long_side"))
            continue
        for level, occupancy in enumerate(settings["target_occupancies"]):
            side = math.ceil(long_side / occupancy)
            lower = np.ceil(np.maximum(box[2:] - side, 0)).astype(int)
            upper = np.floor(np.minimum(box[:2], [width - side, height - side])).astype(int)
            if side > min(height, width) or np.any(lower > upper):
                skipped.append((anchor, occupancy, "crop_does_not_fit"))
                continue
            for _ in range(50 if split == "train" else 1):
                if split == "train":
                    x, y = (rng.randint(int(a), int(b)) for a, b in zip(lower, upper))
                else:
                    x, y = np.clip(
                        np.rint((box[:2] + box[2:] - side) / 2).astype(int),
                        lower,
                        upper,
                    )
                overlap = (boxes[:, 0] < x + side) & (boxes[:, 2] > x)
                overlap &= (boxes[:, 1] < y + side) & (boxes[:, 3] > y)
                complete = (boxes[:, :2] >= np.array([x, y]) - 1e-4).all(axis=1)
                complete &= (boxes[:, 2:] <= np.array([x + side, y + side]) + 1e-4).all(axis=1)
                if np.all(complete[overlap]):
                    break
            else:
                skipped.append((anchor, occupancy, "would_truncate_another_target"))
                continue
            tag = f"obj{anchor}_s{level}"
            windows.append((int(x), int(y), side, anchor, tag, occupancy, long_side))
            if rng.random() < settings["background_ratio"]:
                for _ in range(50):
                    x, y = rng.randint(0, width - side), rng.randint(0, height - side)
                    overlap = (boxes[:, 0] < x + side) & (boxes[:, 2] > x)
                    overlap &= (boxes[:, 1] < y + side) & (boxes[:, 3] > y)
                    if not overlap.any():
                        windows.append((x, y, side, -1, f"{tag}_bg", occupancy, ""))
                        break
                else:
                    skipped.append((anchor, occupancy, "no_background_without_targets"))
    return windows, skipped


def save_crop_preview(output, split, samples):
    """Draw a fixed-size contact sheet of real generated crops and their YOLO labels."""
    tile = 224
    canvas = np.full((4 * (tile + 24), len(samples) * tile, 3), 245, dtype=np.uint8)
    for column, (occupancy, records) in enumerate(samples.items()):
        for row, record in enumerate(records):
            path = output / record[2]
            image = cv2.resize(cv2.imread(str(path)), (tile, tile))
            labels = output / "labels" / split / f"{path.stem}.txt"
            for line in labels.read_text().splitlines():
                _, x, y, width, height = map(float, line.split())
                first = (
                    round((x - width / 2) * tile),
                    round((y - height / 2) * tile),
                )
                second = (
                    round((x + width / 2) * tile),
                    round((y + height / 2) * tile),
                )
                cv2.rectangle(image, first, second, (0, 220, 0), 1)
            panel = canvas[row * (tile + 24) : (row + 1) * (tile + 24), column * tile : (column + 1) * tile]
            panel[24:] = image
            text = f"{split} target={occupancy:.0%} actual={float(record[9]):.1%}" if record[7] else "background"
            cv2.putText(panel, text, (3, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (20, 20, 20), 1)
    if not cv2.imwrite(str(output / f"samples_{split}.jpg"), canvas):
        raise OSError(f"Cannot write preview: {output}")


def prepare_local(config, settings, data):
    """Crop the cleaned training/validation images and preserve source coordinates in a manifest."""
    from ultralytics.utils.ops import xywh2xyxy, xyxy2xywh

    source_data = yaml.safe_load((config.parent / settings["source_data"]).read_text(encoding="utf-8"))
    source = Path(source_data["path"]).expanduser().resolve()
    output = Path(data["path"]).expanduser().resolve()
    size, ratio = settings["crop_size"], settings["background_ratio"]
    occupancies = settings.get("target_occupancies", [])
    if not isinstance(size, int) or size < 1 or not 0 <= ratio <= 1 or data["names"] != source_data["names"]:
        raise ValueError("Check crop_size, background_ratio and matching dataset classes")
    if "target_occupancies" in settings and (
        not occupancies
        or len(set(occupancies)) != len(occupancies)
        or not all(0 < value < 1 for value in occupancies)
        or settings["min_source_long_side"] <= 0
    ):
        raise ValueError("Require unique target_occupancies in (0,1) and positive min_source_long_side")
    splits = yaml.safe_load((source / "splits.yaml").read_text(encoding="utf-8"))
    if set(splits["train"]) & set(splits["val"]):
        raise ValueError("Training and validation videos must be disjoint")
    if occupancies and set(splits["test"]) & (set(splits["train"]) | set(splits["val"])):
        raise ValueError("Test videos must be disjoint from training and validation")
    inputs = {split: sorted((source / source_data[split]).glob("*.jpg")) for split in ("train", "val")}
    for split, paths in inputs.items():
        labels = list((source / "labels" / split).glob("*.txt"))
        if not paths or {p.stem for p in paths} != {p.stem for p in labels}:
            raise ValueError(f"Missing source image/label pairs: {split}")
        if occupancies and any(p.stem.rsplit("_", 1)[0] not in splits[split] for p in paths):
            raise ValueError(f"Source images do not match the video split: {split}")
    image_dirs = {split: f"images/{split}" if occupancies else data[split] for split in inputs}
    output.mkdir(parents=True, exist_ok=False)
    for split in inputs:
        (output / image_dirs[split]).mkdir(parents=True)
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
        windows, skipped = [], []
        if occupancies:
            windows, skipped = multiscale_windows(boxes, image.shape, settings, split, rng)
        else:
            for index, box in enumerate(boxes):
                x = min(max(round((box[0] + box[2] - size) / 2), 0), width - size)
                y = min(max(round((box[1] + box[3] - size) / 2), 0), height - size)
                windows.append((x, y, size, index, f"obj{index}", None, None))
            if not len(boxes) or rng.random() < ratio:
                for _ in range(50):
                    x, y = rng.randint(0, width - size), rng.randint(0, height - size)
                    overlap = (boxes[:, 0] < x + size) & (boxes[:, 2] > x)
                    overlap &= (boxes[:, 1] < y + size) & (boxes[:, 3] > y)
                    if not overlap.any():
                        windows.append((x, y, size, -1, "bg", None, None))
                        break
                else:
                    raise ValueError(f"Cannot sample a background crop without annotated objects: {path}")
        records = []
        for x, y, side, anchor, tag, target, native in windows:
            clipped = (boxes - np.array([x, y, x, y])).clip(0, side)
            keep = (clipped[:, 2] > clipped[:, 0]) & (clipped[:, 3] > clipped[:, 1])
            normalized = xyxy2xywh(clipped[keep]) / side
            text = "\n".join(
                f"{int(cls)} " + " ".join(f"{value:.8f}" for value in box)
                for cls, box in zip(rows[keep, 0], normalized)
            )
            name = f"{path.stem}_{tag}"
            destination = output / image_dirs[split] / f"{name}.jpg"
            if not cv2.imwrite(str(destination), image[y : y + side, x : x + side]):
                raise OSError(f"Cannot write crop: {destination}")
            (output / "labels" / split / f"{name}.txt").write_text(text, encoding="utf-8")
            record = [
                split,
                str(path),
                str(destination.relative_to(output)),
                x,
                y,
                side,
                anchor,
                int(keep.sum()),
            ]
            if occupancies:
                record.extend([target, native / side if anchor >= 0 else "", native])
            records.append(record)
        return records, skipped

    summary = {
        "source": str(source),
        "settings": settings,
        "source_videos": splits,
        "splits": {},
    }
    with ExitStack() as stack:
        stream = stack.enter_context((output / "manifest.csv").open("w", newline="", encoding="utf-8"))
        writer = csv.writer(stream)
        header = [
            "split",
            "source_image",
            "crop_image",
            "x",
            "y",
            "size",
            "anchor",
            "objects",
        ]
        if occupancies:
            header += [
                "target_long_side_ratio",
                "actual_long_side_ratio",
                "source_long_side_px",
            ]
            skipped_writer = csv.writer(stack.enter_context((output / "skipped.csv").open("w", newline="")))
            skipped_writer.writerow(["split", "source_image", "anchor", "target_long_side_ratio", "reason"])
            large_val = stack.enter_context((output / "val_large.txt").open("w"))
        writer.writerow(header)
        for split, paths in inputs.items():
            stats = {
                "source_images": len(paths),
                "images": 0,
                "objects": 0,
                "background_images": 0,
            }
            skipped_counts, distribution = Counter(), {}
            samples = {target: [] for target in occupancies}
            sample_counts = Counter()
            sample_rng = random.Random(f"{settings['seed']}:{split}:preview")
            with ThreadPoolExecutor(max_workers=settings["workers"]) as pool:
                for count, (records, skipped) in enumerate(pool.map(crop_image, ((split, p) for p in paths)), 1):
                    writer.writerows(records)
                    stats["images"] += len(records)
                    stats["objects"] += sum(row[7] for row in records)
                    stats["background_images"] += sum(row[7] == 0 for row in records)
                    if occupancies:
                        for anchor, target, reason in skipped:
                            skipped_writer.writerow([split, str(paths[count - 1]), anchor, target, reason])
                            skipped_counts[reason] += 1
                        for record in records:
                            target = record[8]
                            if split == "val" and target >= 0.6:
                                large_val.write(f"{output / record[2]}\n")
                            if record[7]:
                                values = distribution.setdefault(str(target), [])
                                values.append(record[9])
                                sample_counts[target] += 1
                                slot = sample_rng.randrange(sample_counts[target])
                                if len(samples[target]) < 4:
                                    samples[target].append(record)
                                elif slot < 4:
                                    samples[target][slot] = record
                    if count % 2000 == 0 or count == len(paths):
                        print(
                            f"{split}: {count}/{len(paths)} source images, {stats['images']} crops",
                            flush=True,
                        )
            if occupancies:
                stats["skipped"] = dict(skipped_counts)
                stats["occupancies"] = {
                    target: {
                        "images": len(values),
                        "min": min(values),
                        "max": max(values),
                        "mean": float(np.mean(values)),
                    }
                    for target, values in distribution.items()
                }
                save_crop_preview(output, split, samples)
            summary["splits"][split] = stats
    if occupancies:
        (output / "splits.yaml").write_text(yaml.safe_dump(splits, sort_keys=False), encoding="utf-8")
        (output / "dataset.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        validation = {
            "path": str(output),
            "train": "images/train",
            "val": "val_large.txt",
            "names": data["names"],
        }
        (output / "validation_large.yaml").write_text(yaml.safe_dump(validation, sort_keys=False), encoding="utf-8")
        summary["mixed_inputs"] = {split: data[split] for split in inputs}
    summary["validation"] = "Ground-truth-centered crops for local training monitoring only, not end-to-end metrics."
    summary["test"] = "No cropped test set; use original full frames/videos for final evaluation."
    (output / "summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")
    print(yaml.safe_dump(summary["splits"], sort_keys=False), flush=True)


def main():
    """Prepare disjoint video splits while preserving the original videos and XML annotations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/prepare_local.yaml",
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
                        box = (
                            (xmin + xmax) / 2 - 1,
                            (ymin + ymax) / 2 - 1,
                            xmax - xmin,
                            ymax - ymin,
                        )
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
