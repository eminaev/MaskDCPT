import os
import random
import sys

import numpy as np
from hamcrest import assert_that, equal_to

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from rtts_bootstrap import differences


def records(generator, count):
    images = []
    for _ in range(count):
        size = int(generator.integers(1, 6))
        images.append({
            "tp": generator.random((size, 10)) > 0.4,
            "conf": generator.random(size),
            "pred_cls": generator.choice([0.0, 2.0, 5.0], size),
            "target_cls": generator.choice([0.0, 2.0, 5.0], int(generator.integers(1, 6))),
        })
    return images


def test_parallel_rounds_reproduce_sequential_rounds_exactly():
    generator = np.random.default_rng(random.randint(0, 10**6))
    base, treated = records(generator, 40), records(generator, 40)
    seed = random.randint(0, 10**6)
    assert_that(differences(base, treated, 12, seed, random.randint(2, 4)), equal_to(differences(base, treated, 12, seed, 1)), "parallel bootstrap does not reproduce the sequential one")
