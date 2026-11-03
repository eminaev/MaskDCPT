import torch
from torch import nn


class FeatureLoss(nn.Module):
    """Gap between student and teacher feature maps at matching scales, bridged by learned pointwise adapters that predict teacher features rescaled to unit magnitude."""

    def __init__(self, sources, targets):
        super().__init__()
        self.adapters = nn.ModuleList([nn.Conv2d(source, target, 1) for source, target in zip(sources, targets)])

    def forward(self, students, teachers):
        return torch.stack([
            (adapter(student) - teacher.detach() / teacher.detach().abs().mean().clamp(min=1e-6)).abs().mean()
            for adapter, student, teacher in zip(self.adapters, students, teachers)
        ]).mean()
