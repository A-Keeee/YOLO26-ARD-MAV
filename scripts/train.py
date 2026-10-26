"""Train global or local YOLO26n on ARD-MAV; --dry-run only previews the selected configuration."""

import argparse
from pathlib import Path

import yaml

from ultralytics.models.yolo.detect import DetectionTrainer


class ARDMAVTrainer(DetectionTrainer):
    """Honor the configured total epoch count when resuming ARD-MAV detection training."""

    def check_resume(self, overrides):
        """Restore checkpoint settings and apply the requested total training epochs."""
        super().check_resume(overrides)
        if self.resume:
            self.args.epochs = overrides.get("epochs", self.args.epochs)

    def resume_training(self, ckpt):
        """Reject completed epoch budgets before restoring the checkpoint's training state."""
        if self.resume and ckpt and self.epochs <= ckpt.get("epoch", -1) + 1:
            raise ValueError(f"Checkpoint already reached the requested total of {self.epochs} epochs")
        super().resume_training(ckpt)


def main():
    """Load the separate training configuration and apply optional key=value overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/train_global.yaml"
    )
    parser.add_argument("--dry-run", action="store_true", help="Print settings without loading weights or training")
    parser.add_argument("overrides", nargs="*", help="Optional overrides, e.g. epochs=200 batch=8 imgsz=640")
    args = parser.parse_args()

    from ultralytics.cfg import get_cfg, parse_key_value_pair

    config = args.config.expanduser().resolve()
    settings = yaml.safe_load(config.read_text(encoding="utf-8"))
    settings.update(dict(parse_key_value_pair(pair) for pair in args.overrides))
    for key in ("data", "project"):
        settings[key] = str((config.parent / Path(settings[key]).expanduser()).resolve())
    get_cfg(overrides=settings)
    print(yaml.safe_dump(settings, allow_unicode=True, sort_keys=False))
    if args.dry_run:
        return

    from ultralytics import YOLO

    YOLO(settings.pop("model")).train(trainer=ARDMAVTrainer, **settings)


if __name__ == "__main__":
    main()
