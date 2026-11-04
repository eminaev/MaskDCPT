import json
import os
import sys

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from ultralytics import YOLO

from enhance import OPERATORS

matplotlib.use("Agg")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "savefig.dpi": 300})
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PICKS = os.path.join(ROOT, "results/yolo/qualitative.json")
SETS = {
    "rtts": {"base": "datasets/rtts/hazy", "treated": "datasets/rtts/restored_m", "list": None, "classes": [0, 1, 2, 3, 5]},
    "night": {"base": "datasets/bdd100k/original", "treated": "datasets/bdd100k/restored_m", "list": "night.txt", "classes": [0, 1, 2, 3, 5, 6, 7, 9]},
    "snow": {"base": "datasets/bdd100k/original", "treated": "datasets/bdd100k/clahe", "list": "snowy.txt", "classes": [0, 1, 2, 3, 5, 6, 7, 9]},
}
NAMES = {"clahe": "CLAHE", "gamma": "Gamma", "dcp": "DCP", "retinex": "Retinex", "msrcr": "MSRCR"}


def stems(spec):
    folder = os.path.join(ROOT, spec["base"], "images")
    if spec["list"] is None:
        return sorted(os.path.splitext(name)[0] for name in os.listdir(folder))
    return [os.path.splitext(os.path.basename(line.strip()))[0] for line in open(os.path.join(ROOT, spec["base"], spec["list"])) if line.strip()]


def picture(folder, stem):
    return next(os.path.join(ROOT, folder, "images", stem + suffix) for suffix in [".jpg", ".png"] if os.path.exists(os.path.join(ROOT, folder, "images", stem + suffix)))


def truth(folder, stem, shape):
    text = open(os.path.join(ROOT, folder, "labels", stem + ".txt")).read().split()
    rows = np.array(text, dtype=float).reshape(-1, 5)
    height, width = shape
    return rows[:, 0].astype(int), np.stack([(rows[:, 1] - rows[:, 3] / 2) * width, (rows[:, 2] - rows[:, 4] / 2) * height, (rows[:, 1] + rows[:, 3] / 2) * width, (rows[:, 2] + rows[:, 4] / 2) * height], axis=1)


def overlap(left, right):
    top = np.maximum(left[:, None, :2], right[None, :, :2])
    bottom = np.minimum(left[:, None, 2:], right[None, :, 2:])
    inter = np.prod(np.clip(bottom - top, 0, None), axis=2)
    area = lambda boxes: np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
    return inter / (area(left)[:, None] + area(right)[None, :] - inter + 1e-9)


def matching(kinds, boxes, labels, targets):
    hit, found = np.zeros(len(boxes), bool), np.zeros(len(targets), bool)
    scores = overlap(boxes, targets) if len(boxes) and len(targets) else np.zeros((len(boxes), len(targets)))
    for index in range(len(boxes)):
        options = [column for column in np.argsort(-scores[index]) if scores[index, column] >= 0.5 and not found[column] and labels[column] == kinds[index]]
        if options:
            hit[index], found[options[0]] = True, True
    return hit, found


def detection(model, path, classes):
    result = model.predict(path, conf=0.25, classes=classes, verbose=False)[0]
    return result.boxes.cls.cpu().numpy().astype(int), result.boxes.xyxy.cpu().numpy()


def outcome(model, folder, stem, classes):
    path = picture(folder, stem)
    kinds, boxes = detection(model, path, classes)
    labels, targets = truth(folder, stem, cv2.imread(path).shape[:2])
    hit, found = matching(kinds, boxes, labels, targets)
    return {"kinds": kinds, "boxes": boxes, "targets": targets, "hit": hit, "found": found}


def select():
    model = YOLO(os.path.join(ROOT, "pretrained_models/yolov8s.pt"))
    picks = {}
    for name, spec in SETS.items():
        gains = []
        for stem in stems(spec):
            base, treated = outcome(model, spec["base"], stem, spec["classes"]), outcome(model, spec["treated"], stem, spec["classes"])
            gains.append((int(treated["hit"].sum() - base["hit"].sum()), int((~base["hit"]).sum() - (~treated["hit"]).sum()), len(base["targets"]), stem))
        gains = sorted((gain for gain in gains if 0 < gain[2] <= 30), reverse=name != "night")
        picks[name] = {"stem": gains[0][3], "gain": gains[0][0], "fewer_false": gains[0][1], "objects": gains[0][2], "images": len(gains), "improved": sum(gain[0] > 0 for gain in gains), "worsened": sum(gain[0] < 0 for gain in gains)}
        print(name, picks[name], flush=True)
    json.dump(picks, open(PICKS, "w"), indent=1)


def appearance(out):
    picks = json.load(open(PICKS))
    rows = [("rtts", "RTTS\nhaze"), ("snow", "BDD100K\nsnow"), ("night", "BDD100K\nnight")]
    columns = ["None", *NAMES.values(), "M"]
    figure, axes = plt.subplots(len(rows), len(columns), figsize=(7.2, 1.95))
    for row, (name, title) in enumerate(rows):
        spec = SETS[name]
        frame = cv2.imread(picture(spec["base"], picks[name]["stem"]))
        restored = cv2.imread(picture("datasets/rtts/restored_m" if name == "rtts" else "datasets/bdd100k/restored_m", picks[name]["stem"]))
        images = [frame, *(OPERATORS[key](frame) for key in NAMES), restored]
        for column, image in enumerate(images):
            axis = axes[row, column]
            axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(columns[column], fontsize=8)
            if column == 0:
                axis.set_ylabel(title, fontsize=7)
    figure.subplots_adjust(wspace=0.03, hspace=0.05)
    figure.savefig(os.path.join(out, "fig7_appearance.png"), bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)


def drawing(axis, folder, stem, state, title):
    axis.imshow(cv2.cvtColor(cv2.imread(picture(folder, stem)), cv2.COLOR_BGR2RGB))
    for box, hit in zip(state["boxes"], state["hit"]):
        axis.add_patch(Rectangle(box[:2], *(box[2:] - box[:2]), fill=False, linewidth=0.8, edgecolor="#2ca02c" if hit else "#d62728"))
    for box, found in zip(state["targets"], state["found"]):
        if not found:
            axis.add_patch(Rectangle(box[:2], *(box[2:] - box[:2]), fill=False, linewidth=0.8, linestyle=(0, (2, 1)), edgecolor="#ffd92f"))
    axis.set_title(f"{title}\nTP {int(state['hit'].sum())}, FP {int((~state['hit']).sum())}, missed {int((~state['found']).sum())}", fontsize=7)
    axis.set_xticks([])
    axis.set_yticks([])


def detections(out):
    picks = json.load(open(PICKS))
    models = {name: YOLO(os.path.join(ROOT, f"pretrained_models/{name.lower()}.pt")) for name in ["YOLOv8s", "YOLO26s"]}
    rows = [("rtts", "RTTS\nhaze"), ("night", "BDD100K\nnight")]
    figure, axes = plt.subplots(len(rows), 4, figsize=(7.2, 2.75))
    for row, (name, title) in enumerate(rows):
        spec, stem = SETS[name], picks[name]["stem"]
        for column, (detector, folder, label) in enumerate([(detector, folder, label) for detector in models for folder, label in [(spec["base"], "original"), (spec["treated"], "+ M")]]):
            drawing(axes[row, column], folder, stem, outcome(models[detector], folder, stem, spec["classes"]), f"{detector} {label}")
        axes[row, 0].set_ylabel(title, fontsize=7)
    figure.subplots_adjust(wspace=0.04, hspace=0.32)
    figure.savefig(os.path.join(out, "fig8_detections.png"), bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)


if __name__ == "__main__":
    select() if sys.argv[1] == "select" else {"appearance": appearance, "detections": detections}[sys.argv[1]](sys.argv[2])
