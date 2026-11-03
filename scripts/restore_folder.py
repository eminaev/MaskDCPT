import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from basicsr.archs.nafnet_mobile_arch import NAFNetMobile


def network(path, device):
    checkpoint = torch.load(path, map_location="cpu")
    net = NAFNetMobile(width=checkpoint["width"])
    net.load_state_dict(checkpoint["params"], strict=True)
    return net.eval().fused().to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="experiments/mobile_v4_feat/nafnet_mobile_M.pth")
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    args = parser.parse_args()
    os.makedirs(args.target, exist_ok=True)
    device = torch.device("cuda")
    net = network(args.weights, device)
    names = sorted(os.listdir(args.source))
    for index, name in enumerate(names, 1):
        image = cv2.imread(os.path.join(args.source, name), cv2.IMREAD_COLOR)
        tensor = torch.from_numpy(np.ascontiguousarray(image[:, :, ::-1])).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
        height, width = tensor.shape[2:]
        padded = F.pad(tensor, (0, (16 - width % 16) % 16, 0, (16 - height % 16) % 16), mode="reflect")
        with torch.no_grad():
            output = net(padded)[:, :, :height, :width].clamp(0, 1)
        cv2.imwrite(os.path.join(args.target, os.path.splitext(name)[0] + ".png"), (output[0].permute(1, 2, 0).cpu().numpy()[:, :, ::-1] * 255.0).round().astype(np.uint8))
        if index % 500 == 0:
            print(f"restored {index}/{len(names)}", flush=True)


if __name__ == "__main__":
    main()
