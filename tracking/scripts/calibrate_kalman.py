"""Calibrate GLAD's relative-velocity filter against XML labels, using real detector observations."""

import argparse
import csv
import itertools
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml
from motion import MotionDetector
from track import RelativeVelocity, search_region, track_video

from ultralytics import YOLO
from ultralytics.utils.files import increment_path


def read_frames(path, dataset):
    """Align one-based video frame IDs with VOC labels; missing XML is unknown, never a negative label."""
    with path.open(newline="") as stream:
        frames = list(csv.DictReader(stream))
    sequence = path.name.removesuffix("_frames.csv")
    for expected, row in enumerate(frames, 1):
        if int(row["frame"]) != expected:
            raise ValueError(f"Non-contiguous frame IDs: {path}")
        xml = dataset / "Annotations" / sequence / f"{sequence}_{expected:04d}.xml"
        row["annotated"] = xml.exists()
        row["ambiguous"] = False
        row["truth"] = None
        if xml.exists():
            annotation = ET.parse(xml).getroot()
            objects = annotation.findall("object")
            row["ambiguous"] = len(objects) > 1
            row["annotated"] = not row["ambiguous"]
            if len(objects) == 1:
                row["truth"] = np.array(
                    [float(objects[0].findtext(f"bndbox/{key}")) - 1 for key in ("xmin", "ymin", "xmax", "ymax")]
                )
        row["box"] = np.array([float(row[key]) for key in ("x1", "y1", "x2", "y2")]) if int(row["detected"]) else None
        transform = json.loads(row["roi_homography"])
        row["transform"] = np.asarray(transform) if transform is not None else None
    return frames


def evaluate(frames, settings, shape, horizons=()):
    """Replay recorded observations without feeding labels into the filter; score predictions before correction."""
    state, lost = None, 0
    errors, missed_errors, inside = [], [], []
    forecasts = {step: [] for step in horizons}
    tp = fp = positives = unknown = ambiguous = 0
    for index, row in enumerate(frames):
        box, truth = row["box"], row["truth"]
        if row["mode"] == "GLOBAL":
            state, lost = None, 0
        else:
            predicted, camera = state.predict(row["transform"])
            roi = np.asarray(search_region(predicted, lost, shape, settings))
            if truth is not None:
                center = (truth[:2] + truth[2:]) / 2
                error = float(np.linalg.norm(predicted - center))
                errors.append(error)
                if lost:
                    missed_errors.append(error)
                inside.append(bool(np.all(center >= roi[:2]) and np.all(center < roi[2:])))
        if box is not None:
            center = (box[:2] + box[2:]) / 2
            if state is None:
                state = RelativeVelocity(center, settings["kalman"])
            else:
                state.correct(center, camera)
            lost = 0
        else:
            lost += 1
        if horizons and box is not None and (index + 1) % 30 == 0:
            forecast = RelativeVelocity(state.center, settings["kalman"])
            forecast.filter.statePost = state.filter.statePost.copy()
            forecast.filter.errorCovPost = state.filter.errorCovPost.copy()
            for step, following in enumerate(frames[index + 1 : index + 1 + max(horizons)], 1):
                if following["mode"] == "GLOBAL":
                    break  # Global rows do not contain a measured ROI homography.
                position, _ = forecast.predict(following["transform"])
                if step in forecasts and following["truth"] is not None:
                    center = (following["truth"][:2] + following["truth"][2:]) / 2
                    forecasts[step].append(float(np.linalg.norm(position - center)))
        if row["annotated"]:
            positives += truth is not None
            matched = False
            if box is not None and truth is not None:
                intersection = np.maximum(0, np.minimum(box[2:], truth[2:]) - np.maximum(box[:2], truth[:2])).prod()
                union = (box[2:] - box[:2]).prod() + (truth[2:] - truth[:2]).prod() - intersection
                matched = intersection / union >= 0.5
            tp += int(matched)
            fp += int(box is not None and not matched)
        else:
            unknown += 1
            ambiguous += row["ambiguous"]
    if not errors:
        raise ValueError("No labeled LOCAL predictions to score; use a longer or different sequence")
    recall, precision = tp / positives if positives else 0, tp / (tp + fp) if tp + fp else 0
    return {
        "prediction_frames": len(errors),
        "prediction_rmse_px": float(np.sqrt(np.mean(np.square(errors)))),
        "prediction_median_px": float(np.median(errors)),
        "prediction_p95_px": float(np.percentile(errors, 95)),
        "after_miss_frames": len(missed_errors),
        "after_miss_rmse_px": float(np.sqrt(np.mean(np.square(missed_errors)))) if missed_errors else None,
        "roi_center_coverage": float(np.mean(inside)),
        "true_positives_iou50": tp,
        "false_positives_iou50": fp,
        "positive_frames": positives,
        "unknown_annotation_frames": unknown,
        "ambiguous_annotation_frames": ambiguous,
        "recall_iou50": recall,
        "precision_iou50": precision,
        "f1_iou50": 2 * precision * recall / (precision + recall) if precision + recall else 0,
        "forecast": {
            str(step): {"frames": len(values), "rmse_px": float(np.sqrt(np.mean(np.square(values))))}
            for step, values in forecasts.items()
            if values
        },
    }


def summarize(sequences, settings, frames, horizons=()):
    """Weight videos equally so a longer sequence cannot dominate the parameter selection."""
    results = {}
    for path in sequences:
        metadata = json.loads(path.with_name(path.name.replace("_frames.csv", "_summary.json")).read_text())
        results[path.stem.removesuffix("_frames")] = evaluate(
            frames[path], settings, (metadata["height"], metadata["width"]), horizons
        )
    summary = {
        "videos": results,
        "macro": {
            key: float(np.mean([video[key] for video in results.values()]))
            for key in ("prediction_rmse_px", "roi_center_coverage", "recall_iou50", "precision_iou50", "f1_iou50")
        },
    }
    if horizons:
        if any(len(video["forecast"]) != len(horizons) for video in results.values()):
            raise ValueError("Insufficient labeled forecast samples; use longer videos")
        summary["macro"]["forecast_rmse_px"] = float(
            np.mean([video["forecast"][str(step)]["rmse_px"] for video in results.values() for step in horizons])
        )
    return summary


def calibrate(paths, settings, calibration, output, models, motion):
    """Search completed baseline traces and validate the chosen parameters with fresh detector inference."""
    baseline = settings["kalman"].copy()
    ratios = list(itertools.product(calibration["process_noise_ratios"], calibration["initial_covariance_ratios"]))
    frames = {path: read_frames(path, calibration["dataset"]) for group in paths.values() for path in group}
    horizons = calibration["forecast_horizons"]
    report = {
        "baseline_traces": {key: [str(path.resolve()) for path in group] for key, group in paths.items()},
        "baseline": {
            key: summarize(group, settings, frames, horizons if key == "train" else ()) for key, group in paths.items()
        },
    }
    candidates = [{"kalman": baseline, "train": report["baseline"]["train"]["macro"]}]
    for q, p in ratios:
        parameters = {"process_noise": q, "measurement_noise": 1.0, "initial_covariance": p}
        metrics = summarize(paths["train"], {**settings, "kalman": parameters}, frames, horizons)
        candidates.append({"kalman": parameters, "train": metrics["macro"]})
    best = min(candidates, key=lambda candidate: candidate["train"]["forecast_rmse_px"])
    settings["kalman"] = best["kalman"]
    (output / "candidate_track.yaml").write_text(yaml.safe_dump(settings, allow_unicode=True, sort_keys=False))
    print(f"Selected on training forecasts only: {best}", flush=True)
    report["candidate"] = {
        key: summarize(group, settings, frames, horizons if key == "train" else ()) for key, group in paths.items()
    }
    report["parameters"] = {"baseline": baseline, "candidate": best["kalman"]}
    report["search"] = candidates
    validation = output / "candidate_val"
    validation.mkdir()
    fresh = []
    for name in calibration["val_videos"]:
        track_video(
            calibration["dataset"] / "videos" / f"{name}.mp4",
            validation,
            models,
            motion,
            {**settings, "kalman": best["kalman"], "max_frames": calibration["max_frames"], "show": False},
        )
        path = validation / f"{name}_frames.csv"
        frames[path] = read_frames(path, calibration["dataset"])
        fresh.append(path)
    report["end_to_end_validation"] = summarize(fresh, settings, frames)
    before = report["baseline"]["val"]["macro"]
    replay = report["candidate"]["val"]["macro"]
    after = report["end_to_end_validation"]["macro"]
    report["recommended"] = bool(
        replay["prediction_rmse_px"] < before["prediction_rmse_px"]
        and after["prediction_rmse_px"] < before["prediction_rmse_px"]
        and after["f1_iou50"] >= before["f1_iou50"]
        and after["recall_iou50"] >= before["recall_iou50"]
        and after["roi_center_coverage"] >= before["roi_center_coverage"]
    )
    report["note"] = (
        "Labels score predictions only; all corrections use actual detector boxes including false positives. "
        "Parameters selected on training videos only. Q/R and P0/R are identifiable, not their common scale. "
        "Training objective averages RMSE equally over videos and forecast horizons without observation corrections. "
        "Fixed-observation replay does not simulate changed crops; candidate validation reruns the full cascade. "
        "These are subset diagnostics, not paper AP or final test results. Original config is never overwritten."
    )
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != "search"}, indent=2), flush=True)
    print(f"Calibration saved to {output}", flush=True)

    return report


def main():
    """Select on training videos, check held-out videos, then rerun the full detector cascade for validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/calibrate_kalman.yaml"
    )
    args = parser.parse_args()
    config = args.config.expanduser().resolve()
    calibration = yaml.safe_load(config.read_text())
    for key in ("track_config", "dataset", "splits", "project"):
        calibration[key] = (config.parent / calibration[key]).resolve()
    split = yaml.safe_load(calibration["splits"].read_text())
    train, val = calibration["train_videos"], calibration["val_videos"]
    if not train or not val or len(set(train + val)) != len(train + val):
        raise ValueError("Require nonempty, disjoint training and validation videos")
    if not set(train) <= set(split["train"]) or not set(val) <= set(split["val"]):
        raise ValueError("Calibration must respect the existing train/val split and exclude all test videos")
    ratios = list(itertools.product(calibration["process_noise_ratios"], calibration["initial_covariance_ratios"]))
    if not ratios or any(min(pair) <= 0 for pair in ratios) or calibration["max_frames"] < 0:
        raise ValueError("Require positive covariance ratios and nonnegative max_frames")
    settings = yaml.safe_load(calibration["track_config"].read_text())
    if not calibration["forecast_horizons"] or not all(
        isinstance(step, int) and 0 < step < settings["local_patience"] for step in calibration["forecast_horizons"]
    ):
        raise ValueError("Forecast horizons must be positive integers below local_patience")
    for key in ("global_weights", "local_weights", "classifier_weights", "source", "project"):
        settings[key] = str((calibration["track_config"].parent / settings[key]).resolve())
    output = increment_path(calibration["project"] / calibration["name"], mkdir=True)
    (output / "baseline_config.yaml").write_text(yaml.safe_dump(settings, sort_keys=False))
    (output / "calibration_config.yaml").write_text(
        yaml.safe_dump(
            {key: str(value) if isinstance(value, Path) else value for key, value in calibration.items()},
            sort_keys=False,
        )
    )
    runtime = {**settings, "max_frames": calibration["max_frames"], "show": False}
    models = [YOLO(settings[key]) for key in ("global_weights", "local_weights")]
    motion = MotionDetector(settings["classifier_weights"], settings["motion"])
    paths = {}
    for group, names in (("train", train), ("val", val)):
        folder = output / f"baseline_{group}"
        folder.mkdir()
        paths[group] = []
        for name in names:
            track_video(calibration["dataset"] / "videos" / f"{name}.mp4", folder, models, motion, runtime)
            paths[group].append(folder / f"{name}_frames.csv")
    calibrate(paths, settings, calibration, output, models, motion)


if __name__ == "__main__":
    main()
