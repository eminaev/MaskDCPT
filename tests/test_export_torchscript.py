import os
import random
import sys

import torch
from hamcrest import assert_that, less_than

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from basicsr.archs.nafnet_mobile_arch import NAFNetMobile
from export_torchscript import export


def test_exported_trace_matches_fused_network_on_other_frame_size(tmp_path):
    width = random.choice([8, 12, 16])
    net = NAFNetMobile(width=width).eval()
    torch.save({"params": net.state_dict(), "width": width}, tmp_path / "net.pth")
    export(str(tmp_path / "net.pth"), str(tmp_path / "net.ts"), 64, 96)
    image = torch.rand(1, 3, 16 * random.randint(2, 5), 16 * random.randint(2, 5))
    with torch.no_grad():
        gap = (torch.jit.load(str(tmp_path / "net.ts"))(image) - net.fused()(image)).abs().max().item()
    assert_that(gap, less_than(1e-4), "exported trace does not reproduce the fused network")
