"""Run GLAD's published global/local detector structure with YOLO26 and no MOT tracker."""

import argparse
import csv
import json
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
import yaml
from motion import MotionDetector, estimate_homography

from ultralytics import YOLO, __version__
from ultralytics.engine.results import Results
from ultralytics.utils.files import increment_path


class RelativeVelocity:
    """Estimate (vx, vy, ax, ay) in pixels per frame, following the state definition in GLAD equations 1-3."""

    def __init__(self, center, settings):
        """Initialize at an observed target; covariance choices are explicit implementation parameters."""
        self.center = np.asarray(center, dtype=np.float32)
        self.filter = cv2.KalmanFilter(4, 2)
        self.filter.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
        )
        self.filter.measurementMatrix = np.eye(2, 4, dtype=np.float32)
        self.filter.processNoiseCov = np.eye(4, dtype=np.float32) * settings["process_noise"]
        self.filter.measurementNoiseCov = np.eye(2, dtype=np.float32) * settings["measurement_noise"]
        self.filter.errorCovPost = np.eye(4, dtype=np.float32) * settings["initial_covariance"]
        self.filter.statePost = np.zeros((4, 1), dtype=np.float32)

    def predict(self, homography):
        """Compensate camera motion and add relative velocity; advance the filter exactly once per frame."""
        camera_center = self.center.copy()
        if homography is not None:
            camera_center = cv2.perspectiveTransform(self.center.reshape(1, 1, 2), homography)[0, 0]
        velocity = self.filter.statePost[:2, 0].copy()  # Eq.3 uses U_(t-1).
        self.filter.predict()
        self.center = camera_center + velocity
        return self.center.copy(), camera_center

    def correct(self, center, camera_center):
        """Measure relative displacement after camera compensation, and correct only on actual detections."""
        center = np.asarray(center, dtype=np.float32)
        self.filter.correct((center - camera_center).reshape(2, 1))
        self.center = center


def search_region(center, lost, shape, settings):
    """Use L = 300 + 4*T_lost from GLAD equation 4; shift image-edge windows inside the original frame."""
    side = min(settings["crop_size"] + lost * settings["roi_growth"], *shape[:2])
    x = int(np.clip(round(center[0] - side / 2), 0, shape[1] - side))
    y = int(np.clip(round(center[1] - side / 2), 0, shape[0] - side))
    return x, y, x + side, y + side


def appearance_detect(model, image, settings, confidence, reference=None):
    """Replace only the appearance detector with YOLO26; retain distinct global/local confidence thresholds."""
    boxes = (
        model.predict(
            image,
            imgsz=settings["imgsz"],
            device=settings["device"],
            conf=confidence,
            max_det=100,
            verbose=False,
            save=False,
            nms=False,
        )[0]
        .boxes.data.cpu()
        .numpy()
        .copy()
    )
    if not len(boxes):
        return None
    if reference is not None:
        distance = np.linalg.norm((boxes[:, :2] + boxes[:, 2:4]) / 2 - reference, axis=1)
        return boxes[np.lexsort((distance, -boxes[:, 4]))[0]].copy()
    return boxes[np.argmax(boxes[:, 4])].copy()


def track_video(source, output, models, motion, settings):
    """Apply one global or local cascade per frame; export detections without fabricated object IDs."""
    cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = (int(cap.get(key)) for key in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
    if not cap.isOpened() or fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise ValueError(f"Cannot read video metadata: {source}")
    writer = cv2.VideoWriter(str(output / f"{source.stem}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Cannot create the MP4 output video")
    state, previous, previous_gray = None, None, None
    lost = frame_id = detected_frames = homography_failures = 0
    modes, calls, sources = Counter(), Counter(), Counter()
    trail = deque(maxlen=30)
    start = time.perf_counter()
    try:
        with (output / f"{source.stem}_frames.csv").open("w", newline="") as log:
            rows = csv.writer(log)
            rows.writerow(
                [
                    "frame",
                    "time_s",
                    "mode",
                    "next_mode",
                    "branch",
                    "detected",
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                    "score",
                    "lost_frames",
                    "roi_xyxy",
                    "predicted_center",
                    "roi_homography_ok",
                    "roi_homography",
                    "motion_called",
                    "motion_homography_ok",
                    "motion_candidates",
                    "motion_pass",
                    "cnn_pass",
                ]
            )
            while not settings["max_frames"] or frame_id < settings["max_frames"]:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_id += 1
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                mode = "GLOBAL" if state is None else "LOCAL"
                roi, predicted, camera_center, reference, homography = None, None, None, None, None
                homography_ok, motion_called = "", False
                image, previous_image = frame, previous
                if mode == "LOCAL":
                    homography, _ = estimate_homography(previous_gray, gray, settings["motion"])
                    homography_ok = homography is not None
                    homography_failures += not homography_ok
                    predicted, camera_center = state.predict(homography)
                    roi = search_region(predicted, lost, frame.shape, settings)
                    x1, y1, x2, y2 = roi
                    image, previous_image = frame[y1:y2, x1:x2], previous[y1:y2, x1:x2]
                    reference = predicted - np.array([x1, y1])
                index = int(mode == "LOCAL")
                branch = "LAD" if index else "GAD"
                calls[branch] += 1
                detection = appearance_detect(
                    models[index], image, settings, settings["local_conf" if index else "global_conf"], reference
                )
                diagnostics = {"homography_ok": "", "candidates": 0, "motion_pass": 0, "cnn_pass": 0}
                if detection is None and previous_image is not None:
                    branch = "LMD" if index else "GMD"
                    calls[branch] += 1
                    motion_called = True
                    detection, diagnostics = motion.detect(previous_image, image, reference)
                if detection is not None:
                    if roi is not None:
                        detection[[0, 2]] += roi[0]
                        detection[[1, 3]] += roi[1]
                    center = (detection[:2] + detection[2:4]) / 2
                    if state is None:
                        state = RelativeVelocity(center, settings["kalman"])
                        trail.clear()
                    else:
                        state.correct(center, camera_center)
                    lost = 0
                    detected_frames += 1
                    sources[branch] += 1
                    trail.append(tuple(center.astype(int)))
                else:
                    branch = "NONE"
                    if mode == "LOCAL":
                        lost += 1
                        if lost >= settings["local_patience"]:
                            state = None  # The next frame searches globally, never both YOLO detectors on this frame.
                            trail.clear()
                next_mode = "GLOBAL" if state is None else "LOCAL"
                modes[mode] += 1
                rows.writerow(
                    [
                        frame_id,
                        round((frame_id - 1) / fps, 6),
                        mode,
                        next_mode,
                        branch,
                        int(detection is not None),
                        *(detection[:5].tolist() if detection is not None else [""] * 5),
                        lost,
                        json.dumps(roi),
                        json.dumps(predicted.tolist() if predicted is not None else None),
                        homography_ok,
                        json.dumps(homography.tolist() if homography is not None else None),
                        int(motion_called),
                        diagnostics["homography_ok"],
                        diagnostics["candidates"],
                        diagnostics["motion_pass"],
                        diagnostics["cnn_pass"],
                    ]
                )
                boxes = detection.reshape(1, 6) if detection is not None else np.empty((0, 6), dtype=np.float32)
                canvas = Results(frame, path=str(source), names=models[0].names, boxes=boxes).plot(line_width=2)
                if trail:
                    cv2.polylines(canvas, [np.asarray(trail, dtype=np.int32)], False, (0, 220, 0), 2)
                if roi is not None:
                    cv2.rectangle(canvas, roi[:2], roi[2:], (255, 255, 255), 1)
                if detection is None and predicted is not None:
                    point = tuple(np.clip(predicted, [0, 0], [width - 1, height - 1]).astype(int))
                    cv2.circle(canvas, point, 5, (0, 160, 255), 1)
                cv2.putText(
                    canvas,
                    f"{mode} | {branch} | frame {frame_id} | lost {lost}",
                    (20, 35),
                    0,
                    0.8,
                    (0, 255, 255),
                    2,
                )
                writer.write(canvas)
                previous, previous_gray = frame, gray
                if frame_id % 100 == 0:
                    print(f"{source.name}: frame {frame_id}, {mode}/{branch}, lost={lost}", flush=True)
                if settings["show"]:
                    cv2.imshow("GLAD structure + YOLO26 (no MOT tracker)", canvas)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    finally:
        cap.release()
        writer.release()
        if settings["show"]:
            cv2.destroyAllWindows()
    if not frame_id:
        raise ValueError(f"No decoded video frames: {source}")
    seconds = time.perf_counter() - start
    summary = {
        "source": str(source),
        "frames": frame_id,
        "source_fps": fps,
        "width": width,
        "height": height,
        "modes": dict(modes),
        "detector_calls": dict(calls),
        "detection_sources": dict(sources),
        "frames_with_detections": detected_frames,
        "roi_homography_failures": homography_failures,
        "elapsed_seconds": round(seconds, 3),
        "processing_fps_including_video_io": round(frame_id / seconds, 3),
        "mot_tracker": None,
        "note": "Published GLAD structure with reconstructed unspecified parameters; not a strict numerical reproduction.",
    }
    (output / f"{source.stem}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    """Load the explicit method parameters and trained weights; do not train or use annotation-derived ROIs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/track_glad.yaml")
    parser.add_argument("--source", type=Path, help="Original video file or directory")
    parser.add_argument("--max-frames", type=int, help="Frames per video; 0 processes the complete video")
    parser.add_argument("--show", action="store_true", default=None, help="Display video; press q to stop this video")
    parser.add_argument("--dry-run", action="store_true", help="Check paths and print settings without inference")
    args = parser.parse_args()
    config = args.config.expanduser().resolve()
    settings = yaml.safe_load(config.read_text(encoding="utf-8"))
    for key, value in (("source", args.source), ("max_frames", args.max_frames), ("show", args.show)):
        if value is not None:
            settings[key] = str(value.resolve()) if isinstance(value, Path) else value
    for key in ("source", "global_weights", "local_weights", "classifier_weights", "project"):
        settings[key] = str((config.parent / Path(settings[key]).expanduser()).resolve())
    for key in ("source", "global_weights", "local_weights", "classifier_weights"):
        if not Path(settings[key]).exists():
            raise FileNotFoundError(f"{key}: {settings[key]}")
    if not 0 < settings["local_conf"] <= settings["global_conf"] <= 1:
        raise ValueError("Require 0 < local_conf <= global_conf <= 1")
    if min(settings["crop_size"], settings["local_patience"]) < 1:
        raise ValueError("Require positive crop_size and local_patience")
    if min(settings["roi_growth"], settings["max_frames"]) < 0:
        raise ValueError("roi_growth and max_frames must be nonnegative")
    source = Path(settings["source"])
    videos = (
        sorted(p for p in source.iterdir() if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"})
        if source.is_dir()
        else [source]
    )
    if not videos:
        raise ValueError(f"No videos found: {source}")
    print(yaml.safe_dump(settings, allow_unicode=True, sort_keys=False), flush=True)
    if args.dry_run:
        return
    models = [YOLO(settings[key]) for key in ("global_weights", "local_weights")]
    if models[0].names != models[1].names or len(models[0].names) != 1 or any(m.task != "detect" for m in models):
        raise ValueError("Expected two single-class ARD-MAV detection models with matching names")
    motion = MotionDetector(settings["classifier_weights"], settings["motion"])
    output = increment_path(Path(settings["project"]) / settings["name"], mkdir=True)
    (output / "config.yaml").write_text(
        yaml.safe_dump(
            {**settings, "ultralytics_version": __version__, "mot_tracker": None}, allow_unicode=True, sort_keys=False
        ),
        encoding="utf-8",
    )
    for video in videos:
        track_video(video, output, models, motion, settings)
    print(f"Results saved to {output}", flush=True)


if __name__ == "__main__":
    main()
