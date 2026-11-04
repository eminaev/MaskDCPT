import os
import random
import sys

import numpy as np
from hamcrest import assert_that, equal_to

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from qualitative import matching, tally


def boxes(generator, count):
    corner = generator.uniform(0, 500, (count, 2))
    return np.concatenate([corner, corner + generator.uniform(20, 80, (count, 2))], axis=1)


def test_shifted_prediction_of_same_class_matches_its_target():
    generator = np.random.default_rng(random.randint(0, 10**6))
    targets = boxes(generator, 1)
    hit, found = matching(np.array([3]), targets + generator.uniform(-2, 2, (1, 4)), np.array([3]), targets)
    assert_that((hit.tolist(), found.tolist()), equal_to(([True], [True])), "a nearly identical box of the same class does not match")


def test_prediction_of_other_class_does_not_match_its_target():
    generator = np.random.default_rng(random.randint(0, 10**6))
    targets = boxes(generator, 1)
    hit, found = matching(np.array([2]), targets, np.array([5]), targets)
    assert_that((hit.tolist(), found.tolist()), equal_to(([False], [False])), "a box of another class matches")


def test_two_predictions_cannot_claim_one_target():
    generator = np.random.default_rng(random.randint(0, 10**6))
    targets = boxes(generator, 1)
    hit, _ = matching(np.array([0, 0]), np.concatenate([targets, targets]), np.array([0]), targets)
    assert_that(int(hit.sum()), equal_to(1), "one target is matched twice")


def test_tally_counts_images_where_true_positives_rose_and_fell():
    rises, falls, ties = random.randint(1, 9), random.randint(1, 9), random.randint(0, 9)
    gains = [(random.randint(1, 5), 0, 3, "a")] * rises + [(-random.randint(1, 5), 0, 3, "b")] * falls + [(0, 0, 3, "c")] * ties
    assert_that(tally(gains), equal_to({"images": rises + falls + ties, "improved": rises, "worsened": falls}), "tally miscounts the images")
