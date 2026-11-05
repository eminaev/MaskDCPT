import json
import os
import random
import sys

from hamcrest import assert_that, close_to, contains_exactly

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from graphical_abstract import gap


def test_gap_reads_bootstrap_difference_in_percentage_points(tmp_path):
    low, mean, high = sorted(random.uniform(-0.2, 0.2) for _ in range(3))
    (tmp_path / "result.json").write_text(json.dumps({"gap": {"all": {"mean": mean, "low": low, "high": high, "share_not_positive": 0.5}}}))
    assert_that(gap(str(tmp_path / "result.json")), contains_exactly(close_to(100 * mean, 1e-9), close_to(100 * low, 1e-9), close_to(100 * high, 1e-9)), "gap does not convert the bootstrap interval to points")
