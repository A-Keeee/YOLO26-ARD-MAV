"""Track ARD-MAV videos with Ultralytics' built-in trackers."""

import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    """Save an annotated tracking video and per-frame YOLO labels with confidence and track IDs."""
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        type=Path,
        default=root / "runs/yolo26n_local_multiscale_ft_50e2/weights/best.pt",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/home/ake/SJTU/graduation-project/dataset/ARD-MAV/videos/phantom12.mp4"),
    )
    parser.add_argument(
        "--tracker",
        default="botsort.yaml",
        help="Built-in tracker YAML name or custom YAML path",
    )
    parser.add_argument(
        "--imgsz", type=int, default=1920, help="Inference size; retain small UAVs in full-resolution frames"
    )
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--device", default="0", help="CUDA device index or cpu")
    parser.add_argument("--project", type=Path, default=root / "runs/track")
    parser.add_argument("--show", action="store_true", help="Display tracking results while processing")
    args = parser.parse_args()
    for path in (args.weights, args.source):
        if not path.is_file():
            raise FileNotFoundError(path)

    model = YOLO(args.weights)
    for _ in model.track(
        source=str(args.source),
        tracker=args.tracker,
        stream=True,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        project=str(args.project),
        name=f"{args.source.stem}_{Path(args.tracker).stem}",
        save=True,
        save_txt=True,
        save_conf=True,
        show=args.show,
    ):
        pass


if __name__ == "__main__":
    main()
