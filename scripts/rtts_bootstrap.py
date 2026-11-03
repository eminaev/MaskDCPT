import argparse
import json
import os
from multiprocessing import get_context

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


SHARED = {}


def share(base, treated):
    SHARED["base"], SHARED["treated"] = base, treated


def round_scores(picks):
    return score(SHARED["base"], picks), score(SHARED["treated"], picks)


def differences(base, treated, rounds, seed, workers):
    generator = np.random.default_rng(seed)
    draws = [generator.integers(0, len(base), len(base)) for _ in range(rounds)]
    gaps = {kind: [] for kind in [*score(base, np.arange(len(base))), "all"]}
    share(base, treated)
    with get_context("fork").Pool(workers, initializer=share, initargs=(base, treated)) as pool:
        for index, (left, right) in enumerate(pool.imap(round_scores, draws, chunksize=4), 1):
            for kind in left:
                gaps[kind].append(right[kind] - left[kind])
            gaps["all"].append(np.mean(list(right.values())) - np.mean(list(left.values())))
            if index % 100 == 0:
                print(f"round {index}/{rounds}", flush=True)
    return gaps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="pretrained_models/yolov8s.pt")
    parser.add_argument("--base", default="options/local/rtts_hazy.yaml")
    parser.add_argument("--treated", default="options/local/rtts_restored_m.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--rounds", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/yolo/bootstrap.json")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()
    base = statistics(args.weights, args.base, args.imgsz)
    treated = statistics(args.weights, args.treated, args.imgsz)
    if len(base) != len(treated):
        raise ValueError(f"Runs saw {len(base)} and {len(treated)} images, so they cannot be paired")
    if any(not np.array_equal(left["target_cls"], right["target_cls"]) for left, right in zip(base, treated)):
        raise ValueError("Runs visited images in different order, so they cannot be paired")
    everything = np.arange(len(base))
    full = {"base": score(base, everything), "treated": score(treated, everything)}
    gaps = differences(base, treated, args.rounds, args.seed, args.workers)
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
