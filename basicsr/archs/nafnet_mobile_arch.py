import copy

import torch
from torch import nn

from basicsr.utils.registry import ARCH_REGISTRY


class SimpleGate(nn.Module):
    """Gate that scales the first half of the channels by the sigmoid of the second half, so activations grow linearly rather than quadratically."""

    def forward(self, x):
        first, second = x.chunk(2, dim=1)
        return first * torch.sigmoid(second)


class LeanBlock(nn.Module):
    """NAFNet block in deployment form, with normalization and residual scales folded into its convolutions."""

    def __init__(self, conv1, conv2, conv3, sca, conv4, conv5):
        super().__init__()
        self.conv1 = conv1
        self.conv2 = conv2
        self.conv3 = conv3
        self.sca = sca
        self.conv4 = conv4
        self.conv5 = conv5
        self.sg = SimpleGate()

    def forward(self, inp):
        x = self.sg(self.conv2(self.conv1(inp)))
        y = inp + self.conv3(x * self.sca(x))
        return y + self.conv5(self.sg(self.conv4(y)))


class MobileBlock(nn.Module):
    """NAFNet block whose layer normalization is replaced by batch normalization, so it folds away at deployment."""

    def __init__(self, c):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(c)
        self.conv1 = nn.Conv2d(c, 2 * c, 1)
        self.conv2 = nn.Conv2d(2 * c, 2 * c, 3, padding=1, groups=2 * c)
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c, 1))
        self.conv3 = nn.Conv2d(c, c, 1)
        self.norm2 = nn.BatchNorm2d(c)
        self.conv4 = nn.Conv2d(c, 2 * c, 1)
        self.conv5 = nn.Conv2d(c, c, 1)
        self.sg = SimpleGate()
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)))

    def forward(self, inp):
        x = self.sg(self.conv2(self.conv1(self.norm1(inp))))
        y = inp + self.conv3(x * self.sca(x)) * self.beta
        return y + self.conv5(self.sg(self.conv4(self.norm2(y)))) * self.gamma

    def fused(self):
        return LeanBlock(
            self.absorbed(self.norm1, self.conv1),
            copy.deepcopy(self.conv2),
            self.scaled(self.conv3, self.beta),
            copy.deepcopy(self.sca),
            self.absorbed(self.norm2, self.conv4),
            self.scaled(self.conv5, self.gamma),
        )

    def absorbed(self, norm, conv):
        scale = norm.weight / torch.sqrt(norm.running_var + norm.eps)
        shift = norm.bias - norm.running_mean * scale
        weight = conv.weight.detach()[:, :, 0, 0]
        result = nn.Conv2d(conv.in_channels, conv.out_channels, 1)
        result.weight.data.copy_((weight * scale[None, :])[:, :, None, None])
        result.bias.data.copy_(conv.bias.detach() + weight @ shift)
        return result

    def scaled(self, conv, factor):
        result = copy.deepcopy(conv)
        result.weight.data.mul_(factor.detach().view(-1, 1, 1, 1))
        result.bias.data.mul_(factor.detach().view(-1))
        return result


@ARCH_REGISTRY.register()
class NAFNetMobile(nn.Module):
    """Compact NAFNet U-Net for mobile accelerators: batch normalization instead of layer normalization and shallow stages."""

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=(1, 1, 2, 4), dec_blk_nums=(1, 1, 1, 1), window_size=16):
        super().__init__()
        self.window_size = window_size
        self.intro = nn.Conv2d(img_channel, width, 3, padding=1)
        self.ending = nn.Conv2d(width, img_channel, 3, padding=1)
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[MobileBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2
        self.middle_blks = nn.Sequential(*[MobileBlock(chan) for _ in range(middle_blk_num)])
        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)))
            chan = chan // 2
            self.decoders.append(nn.Sequential(*[MobileBlock(chan) for _ in range(num)]))

    def forward(self, inp):
        x = self.intro(inp)
        skips = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            skips.append(x)
            x = down(x)
        x = self.middle_blks(x)
        for up, decoder, skip in zip(self.ups, self.decoders, skips[::-1]):
            x = decoder(up(x) + skip)
        return self.ending(x) + inp

    def fused(self):
        result = copy.deepcopy(self).eval()
        for stage in [*result.encoders, result.middle_blks, *result.decoders]:
            for index, block in enumerate(stage):
                stage[index] = block.fused()
        return result
