import json
import os
import sys

import cv2
import matplotlib
import matplotlib.pyplot as plt

from enhance import OPERATORS

matplotlib.use("Agg")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False})
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "datasets")
RESULTS = os.path.join(ROOT, "results/yolo")
HAZE = {"YOLOv8s": "bootstrap.json", "YOLO11s": "bootstrap_yolo11s.json", "YOLO26s": "bootstrap_yolo26s.json"}
METHODS = {"CLAHE": "clahe", "Gamma": "gamma", "MSRCR": "msrcr", "DCP": "dcp", "Retinex": "retinex"}
COLORS = {"YOLOv8s": "#1f77b4", "YOLO11s": "#ff7f0e", "YOLO26s": "#2ca02c"}
INK, MUTED, GRID = "#222222", "#666666", "#d9d9d9"


def gap(path):
    interval = json.load(open(path))["gap"]["all"]
    return 100 * interval["mean"], 100 * interval["low"], 100 * interval["high"]


def rows():
    return [
        {
            "title": "A  Classical enhancement",
            "note": "no training, negligible cost",
            "picture": lambda: OPERATORS["clahe"](cv2.imread(f"{DATA}/bdd100k/original/images/b8c3f870-c6c86746.jpg")),
            "caption": "CLAHE, BDD100K snow",
            "label": "YOLOv8s on BDD100K, all 10,000 images",
            "points": [(name, "YOLOv8s", *gap(os.path.join(RESULTS, f"bdd_enhance_{key}_yolov8s_all.json"))) for name, key in METHODS.items()],
            "limits": (-6.0, 1.5),
            "verdict": "No consistent gain;\nDCP, Retinex, MSRCR\nalways harmful",
            "mark": "✗",
        },
        {
            "title": "B  Distilled restoration network",
            "note": "6 M parameters, 14 ms per 720p frame",
            "picture": lambda: cv2.imread(f"{DATA}/rtts/restored_m/images/HazeDr_Google_637.png"),
            "caption": "restored RTTS haze image",
            "label": "NAFNetMobile-M in front of an unchanged detector",
            "points": [("haze", detector, *gap(os.path.join(RESULTS, name))) for detector, name in HAZE.items()] + [("night", detector, *gap(os.path.join(RESULTS, f"bdd_bootstrap_{detector.lower()}_night.json"))) for detector in HAZE],
            "limits": (-4.5, 2.6),
            "verdict": "Helps older YOLOv8s\nunder haze; no gain for\nnewer models; harmful\nat night",
            "mark": "✓",
        },
        {
            "title": "C  Fine-tuning (reference)",
            "note": "needs labeled data and retraining",
            "picture": lambda: cv2.imread(f"{DATA}/bdd100k/original/images/b8c3f870-c6c86746.jpg"),
            "caption": "unchanged input, retrained detector",
            "label": "YOLO11s fine-tuned on BDD100K, single run",
            "points": [("BDD100K", "YOLO11s", 14.5, None, None), ("ACDC", "YOLO11s", 2.9, None, None), ("RTTS", "YOLO11s", -5.4, None, None), ("DAWN", "YOLO11s", -7.1, None, None), ("COCO road8", "YOLO11s", -32.0, None, None)],
            "limits": (-44.0, 20.0),
            "verdict": "Large in-domain gain,\nuneven transfer to\nother datasets",
            "mark": "±",
        },
    ]


def thumbnail(axis, row):
    axis.imshow(cv2.cvtColor(row["picture"](), cv2.COLOR_BGR2RGB))
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color(GRID)
    axis.set_xlabel(row["caption"], fontsize=11, color=MUTED, labelpad=4)
    axis.set_title(row["title"], loc="left", fontsize=15, fontweight="bold", color=INK, pad=24)
    axis.text(0, 1.04, row["note"], transform=axis.transAxes, fontsize=11, color=MUTED, va="bottom")


def chart(axis, row):
    points = row["points"]
    positions = list(range(len(points)))[::-1]
    axis.axvline(0, color="#888888", linewidth=1.2)
    for position, (name, detector, value, low, high) in zip(positions, points):
        color = COLORS[detector]
        if low is None:
            axis.barh(position, value, height=0.55, color=color, edgecolor="white", linewidth=2)
            axis.text(value + (0.8 if value >= 0 else -0.8), position, f"{value:+.1f}".replace("-", "−"), va="center", ha="left" if value >= 0 else "right", fontsize=11, color=INK)
        else:
            axis.plot([low, high], [position, position], color=color, linewidth=2.5, solid_capstyle="round")
            axis.plot(value, position, "o", markersize=9, color=color, markeredgecolor="white", markeredgewidth=2)
    axis.set_yticks(positions)
    axis.set_yticklabels([name if row["title"][0] != "B" else f"{detector}" for name, detector, *_ in points], fontsize=11, color=INK)
    if row["title"][0] == "B":
        axis.axhline(2.5, color=GRID, linewidth=1)
        axis.text(row["limits"][0] + 0.1, 4, "RTTS haze", ha="left", va="center", fontsize=11, color=MUTED)
        axis.text(row["limits"][1] - 0.1, 1, "BDD100K night", ha="right", va="center", fontsize=11, color=MUTED)
    axis.tick_params(axis="y", length=0)
    axis.tick_params(axis="x", labelsize=10, colors=MUTED)
    axis.set_xlim(*row["limits"])
    axis.grid(axis="x", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.set_title(row["label"], loc="left", fontsize=11, color=MUTED, pad=6)
    axis.set_xlabel("Δ mAP50, percentage points" + (" (95% CI)" if row["points"][0][3] is not None else ""), fontsize=10, color=MUTED)


def verdict(axis, row):
    axis.axis("off")
    axis.text(0.0, 0.5, row["mark"], fontsize=30, color=INK, va="center", ha="left", fontweight="bold")
    axis.text(0.2, 0.5, row["verdict"], fontsize=12.5, color=INK, va="center", ha="left", linespacing=1.35)


def main(out):
    figure = plt.figure(figsize=(13.0, 6.6))
    grid = figure.add_gridspec(3, 3, width_ratios=[1.0, 1.55, 0.95], hspace=0.95, wspace=0.28, left=0.03, right=0.99, top=0.77, bottom=0.08)
    for index, row in enumerate(rows()):
        thumbnail(figure.add_subplot(grid[index, 0]), row)
        chart(figure.add_subplot(grid[index, 1]), row)
        verdict(figure.add_subplot(grid[index, 2]), row)
    figure.text(0.03, 0.965, "Improving deployed YOLO detectors in adverse conditions without changing their architecture", fontsize=16.5, fontweight="bold", color=INK, va="top")
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=10, color=color, label=name) for name, color in COLORS.items()]
    figure.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.025, 0.925), ncol=3, frameon=False, fontsize=12, handletextpad=0.3, columnspacing=1.4)
    figure.text(0.99, 0.895, "Validate any preprocessing on the deployment domain:\nvisual quality does not predict detection accuracy", fontsize=12.5, color=INK, ha="right", va="center", style="italic")
    figure.savefig(os.path.join(out, "graphical_abstract.png"), dpi=300, facecolor="white")


if __name__ == "__main__":
    main(sys.argv[1])
