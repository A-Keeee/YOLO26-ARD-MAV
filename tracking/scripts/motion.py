"""GLAD motion cascade from the published method, using the authors' small appearance-classifier checkpoint."""

import cv2
import numpy as np
import torch
from torch import nn


def estimate_homography(previous, current, settings):
    """Track a uniform 30x20 grid with LK and fit a previous-to-current projective transform with RANSAC."""
    height, width = previous.shape
    columns, rows = settings["grid"]
    x = np.linspace(0, width - 1, columns + 2, dtype=np.float32)[1:-1]
    y = np.linspace(0, height - 1, rows + 2, dtype=np.float32)[1:-1]
    points = np.stack(np.meshgrid(x, y), axis=-1).reshape(-1, 1, 2)
    following, status, _ = cv2.calcOpticalFlowPyrLK(
        previous,
        current,
        points,
        None,
        winSize=(15, 15),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if following is None or status is None:
        return None, 0.0
    valid = (status.reshape(-1) == 1) & np.isfinite(following.reshape(-1, 2)).all(axis=1)
    before, after = points.reshape(-1, 2)[valid], following.reshape(-1, 2)[valid]
    if len(before) < settings["min_matches"]:
        return None, 0.0
    homography, inliers = cv2.findHomography(before, after, cv2.RANSAC, settings["ransac_threshold"])
    if homography is None or not np.isfinite(homography).all() or abs(np.linalg.det(homography)) < 1e-8:
        return None, 0.0
    inliers = inliers.reshape(-1).astype(bool)
    if inliers.sum() < settings["min_matches"]:
        return None, 0.0
    background_motion = np.linalg.norm(after[inliers] - before[inliers], axis=1).mean()
    return homography, float(background_motion)


def merge_regions(boxes, distance):
    """Merge foreground rectangles whose minimum edge-to-edge distance is below the paper's D1 threshold."""
    pending = np.asarray(boxes, dtype=np.int32).reshape(-1, 4)
    merged = []
    while len(pending):
        region, pending = pending[0].copy(), pending[1:]
        while len(pending):
            gap = np.maximum(0, np.maximum(region[:2] - pending[:, 2:], pending[:, :2] - region[2:]))
            nearby = np.linalg.norm(gap, axis=1) < distance
            if not nearby.any():
                break
            region = np.r_[
                np.minimum(region[:2], pending[nearby, :2].min(axis=0)),
                np.maximum(region[2:], pending[nearby, 2:].max(axis=0)),
            ]
            pending = pending[~nearby]
        merged.append(region)
    return merged


class AppearanceClassifier(nn.Module):
    """Match the LeNet-shaped state dictionary released as official weights/Net_best.pth."""

    def __init__(self):
        """Build the checkpoint's two-convolution and three-linear-layer architecture."""
        super().__init__()
        self.conv1, self.conv2 = nn.Conv2d(3, 6, 5), nn.Conv2d(6, 16, 5)
        self.fc1, self.fc2, self.fc3 = nn.Linear(400, 120), nn.Linear(120, 84), nn.Linear(84, 2)

    def forward(self, images):
        """Return logits; preprocessing preserves BGR and scales uint8 to [0,1], as in the released code."""
        features = nn.functional.max_pool2d(torch.relu(self.conv1(images)), 2)
        features = nn.functional.max_pool2d(torch.relu(self.conv2(features)), 2)
        features = torch.relu(self.fc1(features.flatten(1)))
        return self.fc3(torch.relu(self.fc2(features)))


class MotionDetector:
    """Apply compensation, frame difference, morphology, motion-feature filtering, and CNN classification."""

    def __init__(self, weights, settings):
        """Load the official tensor-only CNN checkpoint once, without downloading or training at runtime."""
        self.settings = settings
        self.classifier = AppearanceClassifier().eval()
        self.classifier.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)

    def detect(self, previous, current, reference=None):
        """Return one observed MAV proposal in input-image coordinates, or None when the cascade rejects all."""
        settings = self.settings
        diagnostics = {"homography_ok": False, "candidates": 0, "motion_pass": 0, "cnn_pass": 0}
        before = cv2.cvtColor(cv2.GaussianBlur(previous, (11, 11), 0), cv2.COLOR_BGR2GRAY)
        after = cv2.cvtColor(cv2.GaussianBlur(current, (11, 11), 0), cv2.COLOR_BGR2GRAY)
        homography, background_motion = estimate_homography(before, after, settings)
        if homography is None:
            return None, diagnostics  # Unknown alignment must not masquerade as successful motion compensation.
        diagnostics["homography_ok"] = True
        height, width = before.shape
        aligned = cv2.warpPerspective(before, homography, (width, height))
        valid = cv2.warpPerspective(np.full(before.shape, 255, dtype=np.uint8), homography, (width, height))
        valid = cv2.erode(valid, np.ones((3, 3), dtype=np.uint8))
        illumination = float(cv2.absdiff(after, before).mean())
        threshold = (
            settings["difference_threshold"]
            + settings["light_coefficient"] * illumination
            + settings["background_coefficient"] * background_motion
        )
        binary = cv2.threshold(cv2.absdiff(after, aligned), threshold, 255, cv2.THRESH_BINARY)[1]
        binary[valid < 255] = 0
        kernel = np.ones((3, 3), dtype=np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=settings["open_iterations"])
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=settings["close_iterations"])
        _, _, components, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        boxes = []
        for x, y, w, h, area in components[1:]:
            if area >= settings["min_area"]:
                boxes.append([x, y, x + w, y + h])
        regions = merge_regions(boxes, settings["merge_distance"])
        diagnostics["candidates"] = len(regions)
        proposals, patches = [], []
        for box in regions:
            x1, y1 = np.maximum(box[:2] - 2, 0)
            x2, y2 = np.minimum(box[2:] + 2, [width, height])
            first, second = aligned[y1:y2, x1:x2], after[y1:y2, x1:x2]
            corners = cv2.goodFeaturesToTrack(first, maxCorners=30, qualityLevel=0.15, minDistance=3, blockSize=3)
            if corners is None:
                continue
            following, status, _ = cv2.calcOpticalFlowPyrLK(
                first,
                second,
                corners,
                None,
                winSize=(15, 15),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03),
            )
            if following is None or status is None:
                continue
            vectors = (following - corners).reshape(-1, 2)[status.reshape(-1) == 1]
            vectors = vectors[np.isfinite(vectors).all(axis=1)]
            if not len(vectors):
                continue
            speeds = np.linalg.norm(vectors, axis=1)
            angles = np.arctan2(vectors[:, 1], vectors[:, 0])
            # Equation 14 uses variances, unlike std/mean ratios in the incomplete public demo.
            if (
                np.var(angles) > settings["angle_variance"]
                or np.var(speeds) > settings["speed_variance"]
                or speeds.mean() < settings["min_speed"]
            ):
                continue
            proposals.append(box)
            patches.append(cv2.resize(current[y1:y2, x1:x2], (32, 32)))
        diagnostics["motion_pass"] = len(proposals)
        if not proposals:
            return None, diagnostics
        images = torch.from_numpy(np.stack(patches).transpose(0, 3, 1, 2).copy()).float().div_(255)
        with torch.inference_mode():
            probabilities = self.classifier(images).softmax(dim=1).numpy()
        accepted = probabilities.argmax(axis=1) == 1  # Official class index 1 denotes MAV.
        boxes = np.asarray(proposals, dtype=np.float32)[accepted]
        scores = probabilities[accepted, 1]
        diagnostics["cnn_pass"] = len(boxes)
        if not len(boxes):
            return None, diagnostics
        index = (
            np.argmax(scores)
            if reference is None
            else np.argmin(np.linalg.norm((boxes[:, :2] + boxes[:, 2:4]) / 2 - reference, axis=1))
        )
        return np.r_[boxes[index], scores[index], 0].astype(np.float32), diagnostics
