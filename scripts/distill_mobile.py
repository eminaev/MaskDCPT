import argparse
import json
import math
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from basicsr.archs.nafnet_arch import NAFNetBaseline
from basicsr.archs.nafnet_mobile_arch import NAFNetMobile
from basicsr.losses.feature_distill_loss import FeatureLoss


class Pairs(Dataset):
    """Balanced mixture of the five degradations of the 5D benchmark, served as aligned random crops."""

    def __init__(self, root, crop, length):
        self.root = root
        self.crop = crop
        self.length = length
        self.tasks = {
            "noise": [(None, os.path.join(root, "noise/clean", name)) for name in sorted(os.listdir(os.path.join(root, "noise/clean")))],
            "rain": self.matched("rain/input", "rain/target"),
            "haze": [(os.path.join(root, "haze/hazy", name), os.path.join(root, "haze/clear", name.split("_")[0] + ".jpg")) for name in sorted(os.listdir(os.path.join(root, "haze/hazy")))],
            "lowlight": self.matched("lowlight/low", "lowlight/high"),
            "blur": self.matched("blur/input", "blur/target"),
        }

    def matched(self, lq, gt):
        return [(os.path.join(self.root, lq, name), os.path.join(self.root, gt, name)) for name in sorted(os.listdir(os.path.join(self.root, lq)))]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        task = random.choice(list(self.tasks))
        lq, gt = random.choice(self.tasks[task])
        clean = cv2.imread(gt, cv2.IMREAD_COLOR)
        noisy = clean if lq is None else cv2.imread(lq, cv2.IMREAD_COLOR)
        height, width = clean.shape[:2]
        top = random.randint(0, height - self.crop)
        left = random.randint(0, width - self.crop)
        pair = np.stack([noisy[top:top + self.crop, left:left + self.crop], clean[top:top + self.crop, left:left + self.crop]])
        pair = pair[:, :, ::-1] if random.random() < 0.5 else pair
        pair = pair[:, ::-1] if random.random() < 0.5 else pair
        pair = np.rot90(pair, random.randint(0, 3), axes=(1, 2))
        tensor = torch.from_numpy(np.ascontiguousarray(pair[..., ::-1])).permute(0, 3, 1, 2).float() / 255.0
        degraded = tensor[0] + torch.randn_like(tensor[0]) * random.choice([15, 25, 50]) / 255.0 if lq is None else tensor[0]
        return degraded, tensor[1]


def teacher(path, device):
    net = NAFNetBaseline(width=64, enc_blk_nums=[1, 1, 1, 28], middle_blk_num=1, dec_blk_nums=[1, 1, 1, 1], window_size=16)
    net.load_state_dict(torch.load(path, map_location="cpu")["params_ema"], strict=True)
    return net.eval().half().to(device).to(memory_format=torch.channels_last)


def tapped(modules):
    trace = []
    for module in modules:
        module.register_forward_hook(lambda module, inputs, output: trace.append(output))
    return trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="datasets/train")
    parser.add_argument("--teacher", default="pretrained_models/nafnet_maskdcpt_5d.pth")
    parser.add_argument("--out", default="experiments/mobile")
    parser.add_argument("--students", default='{"S": 16, "M": 32}')
    parser.add_argument("--iters", type=int, default=20000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--crop", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--features", type=float, default=0.0)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda")
    source = teacher(args.teacher, device)
    widths = json.loads(args.students)
    students = {name: NAFNetMobile(width=width, enc_blk_nums=[1, 1, 2, 4], middle_blk_num=1, dec_blk_nums=[1, 1, 1, 1]).to(device).to(memory_format=torch.channels_last) for name, width in widths.items()}
    bridges = {name: FeatureLoss([width * 16, width * 8, width * 4, width * 2, width], [1024, 512, 256, 128, 64]).to(device) for name, width in widths.items()}
    optimizers = {name: torch.optim.AdamW([*net.parameters(), *bridges[name].parameters()], lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0) for name, net in students.items()}
    lessons = tapped([source.middle_blks, *[getattr(source, f"decoder{index}") for index in range(4)]])
    answers = {name: tapped([net.middle_blks, *net.decoders]) for name, net in students.items()}
    loader = DataLoader(Pairs(args.data, args.crop, args.iters * args.batch), batch_size=args.batch, num_workers=args.workers, pin_memory=True, drop_last=True, persistent_workers=True, worker_init_fn=lambda worker: random.seed(args.seed * 1000 + worker))
    log = open(os.path.join(args.out, "train.log"), "a")
    print(json.dumps(vars(args)), file=log, flush=True)
    start = time.time()
    running = {name: 0.0 for name in students}
    for step, (degraded, clean) in enumerate(loader, 1):
        degraded = degraded.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        clean = clean.to(device, non_blocking=True)
        lessons.clear()
        with torch.no_grad():
            target = source(degraded.half()).float()
        rate = 1e-6 + 0.5 * (args.lr - 1e-6) * (1 + math.cos(math.pi * step / args.iters))
        for name, net in students.items():
            for group in optimizers[name].param_groups:
                group["lr"] = rate
            answers[name].clear()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = net(degraded).float()
            loss = F.l1_loss(output, target) + F.l1_loss(output, clean) + (args.features * bridges[name]([answer.float() for answer in answers[name]], [lesson.float() for lesson in lessons]) if args.features > 0 else 0.0)
            optimizers[name].zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.1)
            optimizers[name].step()
            running[name] += loss.item()
        if step % 200 == 0 or step == args.iters:
            line = f"step {step}/{args.iters} lr {rate:.2e} " + " ".join(f"{name} {running[name] / 200:.4f}" for name in students) + f" elapsed {(time.time() - start) / 60:.1f} min"
            print(line, flush=True)
            print(line, file=log, flush=True)
            running = {name: 0.0 for name in students}
        if step % 5000 == 0 or step == args.iters:
            for name, net in students.items():
                torch.save({"params": net.state_dict(), "step": step, "width": widths[name]}, os.path.join(args.out, f"nafnet_mobile_{name}.pth"))


if __name__ == "__main__":
    main()
