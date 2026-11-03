import argparse
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from basicsr.archs.nafnet_mobile_arch import NAFNetMobile


def export(weights, out, height, width):
    checkpoint = torch.load(weights, map_location="cpu")
    net = NAFNetMobile(width=checkpoint["width"])
    net.load_state_dict(checkpoint["params"], strict=True)
    with torch.no_grad():
        torch.jit.trace(net.eval().fused(), torch.rand(1, 3, height, width)).save(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="experiments/mobile_v4_feat/nafnet_mobile_M.pth")
    parser.add_argument("--out", default="experiments/mobile_v4_feat/nafnet_mobile_M_fused.ts")
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    args = parser.parse_args()
    export(args.weights, args.out, args.height, args.width)


if __name__ == "__main__":
    main()
