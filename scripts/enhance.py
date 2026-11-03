import argparse
import os
from multiprocessing import Pool

import cv2
import numpy as np


def clahe(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def gamma(frame):
    return cv2.LUT(frame, np.array([round(255 * (level / 255) ** (1 / 1.2)) for level in range(256)], dtype=np.uint8))


def dcp(frame):
    image = frame.astype(np.float64) / 255.0
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dark = cv2.erode(image.min(axis=2), kernel)
    top = np.argsort(dark.ravel())[-max(1, dark.size // 1000):]
    light = np.maximum(image.reshape(-1, 3)[top].max(axis=0), 1e-6)
    transmission = np.maximum(1.0 - 0.95 * cv2.erode((image / light).min(axis=2), kernel), 0.1)
    return np.clip(((image - light) / transmission[:, :, None] + light) * 255.0, 0, 255).round().astype(np.uint8)


def stretched(image, low, high):
    return np.stack([np.clip((channel - np.percentile(channel, low)) / max(np.percentile(channel, high) - np.percentile(channel, low), 1e-6), 0, 1) for channel in np.moveaxis(image, 2, 0)], axis=2)


def surround(image, sigma):
    return np.log(image) - np.log(cv2.GaussianBlur(image, (0, 0), sigma))


def retinex(frame):
    return (stretched(surround(frame.astype(np.float64) + 1.0, 30), 0, 100) * 255.0).round().astype(np.uint8)


def msrcr(frame):
    image = frame.astype(np.float64) + 1.0
    scales = sum(surround(image, sigma) for sigma in [15, 80, 250]) / 3
    color = 46 * (np.log(125 * image) - np.log(image.sum(axis=2, keepdims=True)))
    return (stretched(1.0 * scales * color + 0.0, 1, 99) * 255.0).round().astype(np.uint8)


OPERATORS = {"clahe": clahe, "gamma": gamma, "dcp": dcp, "retinex": retinex, "msrcr": msrcr}


def job(task):
    operator, source, target = task
    cv2.imwrite(target, OPERATORS[operator](cv2.imread(source, cv2.IMREAD_COLOR)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--operator", choices=sorted(OPERATORS), required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()
    os.makedirs(args.target, exist_ok=True)
    tasks = [(args.operator, os.path.join(args.source, name), os.path.join(args.target, os.path.splitext(name)[0] + ".png")) for name in sorted(os.listdir(args.source))]
    with Pool(args.workers) as pool:
        for index, _ in enumerate(pool.imap_unordered(job, tasks, chunksize=8), 1):
            if index % 1000 == 0:
                print(f"{args.operator}: {index}/{len(tasks)}", flush=True)


if __name__ == "__main__":
    main()
