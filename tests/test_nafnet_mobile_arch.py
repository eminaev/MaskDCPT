import io
import random

import onnx
import torch
from hamcrest import assert_that, equal_to, empty, less_than

from basicsr.archs.nafnet_mobile_arch import NAFNetMobile


def test_restores_image_of_same_shape():
    net = NAFNetMobile(width=random.choice([8, 12, 16]), enc_blk_nums=[1, 1, 2, 3], middle_blk_num=1, dec_blk_nums=[1, 1, 1, 1]).eval()
    image = torch.rand(1, 3, 16 * random.randint(2, 6), 16 * random.randint(2, 6))
    assert_that(tuple(net(image).shape), equal_to(tuple(image.shape)), "network does not keep the image shape")


def test_fused_network_matches_original_output():
    net = NAFNetMobile(width=8, enc_blk_nums=[1, 2, 1, 2], middle_blk_num=1, dec_blk_nums=[1, 1, 1, 1])
    net.train()
    with torch.no_grad():
        for _ in range(3):
            net(torch.rand(4, 3, 64, 64) * random.uniform(0.5, 2.0))
        for parameter in net.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.05)
    net.eval()
    image = torch.rand(2, 3, 48, 80)
    with torch.no_grad():
        gap = (net(image) - net.fused()(image)).abs().max().item()
    assert_that(gap, less_than(1e-4), "fused network does not reproduce the original output")


def test_output_stays_finite_for_input_far_outside_training_range():
    net = NAFNetMobile(width=8, enc_blk_nums=[1, 1, 2, 4], middle_blk_num=1, dec_blk_nums=[1, 1, 1, 1])
    with torch.no_grad():
        for parameter in [p for name, p in net.named_parameters() if name.endswith("beta") or name.endswith("gamma")]:
            parameter.fill_(1.0)
    net.eval()
    with torch.no_grad():
        output = net(torch.rand(1, 3, 64, 64) * random.uniform(50.0, 100.0))
    assert_that(bool(torch.isfinite(output).all()), equal_to(True), "network output does not stay finite")


def test_fused_network_has_no_batch_norm():
    net = NAFNetMobile(width=8, enc_blk_nums=[1, 1, 1, 1], middle_blk_num=1, dec_blk_nums=[1, 1, 1, 1]).eval()
    kinds = [type(module).__name__ for module in net.fused().modules() if isinstance(module, torch.nn.BatchNorm2d)]
    assert_that(kinds, empty(), "fused network still contains batch normalization")


def test_fused_network_exports_without_layer_norm_or_transpose():
    net = NAFNetMobile(width=8, enc_blk_nums=[1, 1, 1, 1], middle_blk_num=1, dec_blk_nums=[1, 1, 1, 1]).eval().fused()
    buffer = io.BytesIO()
    torch.onnx.export(net, torch.rand(1, 3, 64, 64), buffer, opset_version=13)
    kinds = sorted({node.op_type for node in onnx.load_from_string(buffer.getvalue()).graph.node} & {"LayerNormalization", "Transpose", "ReduceMean"})
    assert_that(kinds, empty(), "exported graph contains operations that mobile accelerators handle poorly")
