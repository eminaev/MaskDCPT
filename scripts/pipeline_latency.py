import argparse
import json
import os
import random
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO


class Restoration:
    """Traced restoration network that takes a BGR frame from host memory and returns the restored BGR frame to host memory."""

    def __init__(self, path, dtype):
        self.net = torch.jit.load(path, map_location="cuda").to(dtype).eval()
        self.dtype = dtype

    def frame(self, image):
        tensor = torch.from_numpy(image).cuda().permute(2, 0, 1).flip(0).unsqueeze(0).to(self.dtype).div(255.0)
        height, width = image.shape[:2]
        padded = F.pad(tensor, (0, (16 - width % 16) % 16, 0, (16 - height % 16) % 16), mode="reflect")
        with torch.no_grad():
            output = self.net(padded)[:, :, :height, :width].clamp(0, 1)
        return output[0].flip(0).permute(1, 2, 0).mul(255.0).round().byte().cpu().numpy()


def clock(step, frames, warm):
    for frame in frames[:warm] if warm > 0 else frames:
        step(frame)
    spans = []
    for frame in frames:
        torch.cuda.synchronize()
        start = time.perf_counter()
        step(frame)
        torch.cuda.synchronize()
        spans.append((time.perf_counter() - start) * 1000.0)
    return {"median": float(np.median(spans)), "mean": float(np.mean(spans)), "p90": float(np.percentile(spans, 90))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", default="pretrained_models/yolov8s.pt")
    parser.add_argument("--restorer", default="experiments/mobile_v4_feat/nafnet_mobile_M_fused.ts")
    parser.add_argument("--images", default="datasets/rtts/hazy/images")
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--warm", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/yolo/latency.json")
    args = parser.parse_args()
    random.seed(args.seed)
    names = random.sample(sorted(os.listdir(args.images)), args.count)
    sets = {
        "rtts": [cv2.imread(os.path.join(args.images, name), cv2.IMREAD_COLOR) for name in names],
        "720p": [cv2.resize(cv2.imread(os.path.join(args.images, name), cv2.IMREAD_COLOR), (1280, 720)) for name in names[:200]],
    }
    print(json.dumps({"rtts_median_size": [int(np.median([frame.shape[1] for frame in sets["rtts"]])), int(np.median([frame.shape[0] for frame in sets["rtts"]]))]}))
    torch.backends.cudnn.benchmark = True
    detector = YOLO(args.detector)
    summary = {}
    for precision, dtype in [("fp32", torch.float32), ("fp16", torch.float16)]:
        restorer = Restoration(args.restorer, dtype)
        half = dtype == torch.float16
        detect = lambda frame: detector.predict(frame, imgsz=640, quantize=16 if half else 32, device=0, verbose=False)
        for kind, frames in sets.items():
            summary[f"{kind} {precision}"] = {
                "yolo": clock(detect, frames, args.warm),
                "restoration": clock(restorer.frame, frames, args.warm),
                "pipeline": clock(lambda frame: detect(restorer.frame(frame)), frames, args.warm),
            }
            print(kind, precision, json.dumps({stage: round(value["median"], 2) for stage, value in summary[f"{kind} {precision}"].items()}), flush=True)
    json.dump(summary, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
