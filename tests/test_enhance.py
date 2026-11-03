import os
import random
import sys

import numpy as np
from hamcrest import assert_that, close_to, equal_to, greater_than, greater_than_or_equal_to

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from enhance import OPERATORS


def picture(height, width):
    return np.random.default_rng(random.randint(0, 10**6)).integers(0, 256, (height, width, 3), dtype=np.uint8)


def test_every_operator_keeps_shape_and_type_of_frame():
    frame = picture(random.randint(40, 90), random.randint(40, 90))
    kinds = {(OPERATORS[name](frame).shape, OPERATORS[name](frame).dtype.str) for name in OPERATORS}
    assert_that(kinds, equal_to({(frame.shape, frame.dtype.str)}), "an operator does not keep the frame shape and type")


def test_gamma_follows_power_law_with_inverse_exponent():
    level = random.randint(1, 254)
    frame = np.full((8, 8, 3), level, dtype=np.uint8)
    assert_that(int(OPERATORS["gamma"](frame)[0, 0, 0]), close_to(255 * (level / 255) ** (1 / 1.2), 0.5), "gamma does not apply the power 1/1.2")


def test_clahe_raises_contrast_of_flat_frame():
    frame = np.clip(120 + np.random.default_rng(random.randint(0, 10**6)).normal(0, 4, (64, 64, 3)), 0, 255).astype(np.uint8)
    assert_that(float(OPERATORS["clahe"](frame).std()), greater_than(float(frame.std())), "clahe does not raise contrast")


def test_dcp_keeps_frame_whose_dark_channel_is_zero():
    frame = picture(random.randint(40, 70), random.randint(40, 70))
    frame[:, :, random.randint(0, 2)] = 0
    gap = np.abs(OPERATORS["dcp"](frame).astype(int) - frame.astype(int)).max()
    assert_that(int(gap), equal_to(0), "dcp changes a frame without haze")


def test_retinex_stretches_every_channel_to_full_range():
    frame = picture(random.randint(40, 70), random.randint(40, 70))
    ranges = {(int(channel.min()), int(channel.max())) for channel in np.moveaxis(OPERATORS["retinex"](frame), 2, 0)}
    assert_that(ranges, equal_to({(0, 255)}), "retinex does not stretch every channel to 0 and 255")


def test_msrcr_clips_lowest_percent_of_every_channel_to_black():
    frame = picture(random.randint(60, 90), random.randint(60, 90))
    share = min(float((channel == 0).mean()) for channel in np.moveaxis(OPERATORS["msrcr"](frame), 2, 0))
    assert_that(share, greater_than_or_equal_to(0.009), "msrcr does not clip the lowest percent of a channel")
