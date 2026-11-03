import random

import torch
from hamcrest import assert_that, equal_to, less_than, none

from basicsr.losses.feature_distill_loss import FeatureLoss


def test_measures_gap_as_single_number():
    size = random.randint(4, 12)
    loss = FeatureLoss([8, 16], [24, 40])
    value = loss([torch.rand(2, 8, size, size), torch.rand(2, 16, size, size)], [torch.rand(2, 24, size, size), torch.rand(2, 40, size, size)])
    assert_that(tuple(value.shape), equal_to(()), "feature loss does not reduce to a single number")


def test_vanishes_once_adapters_learn_linear_teacher():
    torch.manual_seed(random.randint(0, 1000))
    student = torch.randn(4, 6, 8, 8)
    teacher = torch.nn.functional.conv2d(student, torch.randn(20, 6, 1, 1)) * random.uniform(5.0, 50.0)
    loss = FeatureLoss([6], [20])
    optimizer = torch.optim.Adam(loss.parameters(), lr=0.05)
    for _ in range(600):
        optimizer.zero_grad()
        loss([student], [teacher]).backward()
        optimizer.step()
    assert_that(loss([student], [teacher]).item(), less_than(0.02), "adapters do not learn a linear teacher projection")


def test_sends_no_gradient_into_teacher():
    teacher = torch.rand(1, 12, 6, 6, requires_grad=True)
    FeatureLoss([4], [12])([torch.rand(1, 4, 6, 6, requires_grad=True)], [teacher]).backward()
    assert_that(teacher.grad, none(), "feature loss leaks gradient into the teacher")
