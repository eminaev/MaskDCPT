import argparse
import json

import numpy as np
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils.metrics import ap_per_class

KEYS = ["tp", "conf", "pred_cls", "target_cls"]


class RecordingValidator(DetectionValidator):
    """Detection validator that also keeps the matching statistics of every image, so metrics can be recomputed on resampled image sets."""

    record = []

    def update_metrics(self, preds, batch):
        start = len(self.metrics.stats["tp"])
        super().update_metrics(preds, batch)
        for index in range(start, len(self.metrics.stats["tp"])):
            RecordingValidator.record.append({key: self.metrics.stats[key][index].copy() for key in KEYS})


def statistics(weights, data, imgsz):
    RecordingValidator.record = []
    YOLO(weights).val(data=data, imgsz=imgsz, batch=32, device=0, plots=False, verbose=False, validator=RecordingValidator)
    return RecordingValidator.record


def score(images, picks):
    merged = {key: np.concatenate([images[pick][key] for pick in picks], 0) for key in KEYS}
    result = ap_per_class(merged["tp"], merged["conf"], merged["pred_cls"], merged["target_cls"])
    return dict(zip(result[6].tolist(), result[5][:, 0].tolist()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="pretrained_models/yolov8s.pt")
    parser.add_argument("--base", default="options/local/rtts_hazy.yaml")
    parser.add_argument("--treated", default="options/local/rtts_restored_m.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--rounds", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/yolo/bootstrap.json")
    args = parser.parse_args()
    base = statistics(args.weights, args.base, args.imgsz)
    treated = statistics(args.weights, args.treated, args.imgsz)
    if len(base) != len(treated):
        raise ValueError(f"Runs saw {len(base)} and {len(treated)} images, so they cannot be paired")
    if any(not np.array_equal(left["target_cls"], right["target_cls"]) for left, right in zip(base, treated)):
        raise ValueError("Runs visited images in different order, so they cannot be paired")
    everything = np.arange(len(base))
    full = {"base": score(base, everything), "treated": score(treated, everything)}
    generator = np.random.default_rng(args.seed)
    gaps = {kind: [] for kind in [*full["base"], "all"]}
    for round in range(args.rounds):
        picks = generator.integers(0, len(base), len(base))
        left, right = score(base, picks), score(treated, picks)
        for kind in left:
            gaps[kind].append(right[kind] - left[kind])
        gaps["all"].append(np.mean(list(right.values())) - np.mean(list(left.values())))
        if (round + 1) % 100 == 0:
            print(f"round {round + 1}/{args.rounds}", flush=True)
    summary = {
        "images": len(base),
        "rounds": args.rounds,
        "full": {kind: {key: value for key, value in full[kind].items()} | {"all": float(np.mean(list(full[kind].values())))} for kind in full},
        "gap": {str(kind): {"mean": float(np.mean(values)), "low": float(np.percentile(values, 2.5)), "high": float(np.percentile(values, 97.5)), "share_not_positive": float(np.mean(np.array(values) <= 0))} for kind, values in gaps.items()},
    }
    json.dump(summary, open(args.out, "w"), indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
