import importlib.util
from pathlib import Path

import numpy as np


_p = Path(__file__).parents[1] / "scripts" / "router" / "train_acc_soft_router.py"
_s = importlib.util.spec_from_file_location("train_acc_soft_router", _p)
_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)

def test_normalized_solve_targets_and_all_fail():
    rows = [
        {"per_expert": {"a": 1, "b": 0, "c": 1}},
        {"per_expert": {"a": 0, "b": 0, "c": 0}},
        {"per_expert": {"a": 1, "b": 1, "c": 1}},
    ]
    y, n = _m.normalized_solve_targets(rows, ["a", "b", "c"])
    np.testing.assert_allclose(y[0], [0.5, 0.0, 0.5])
    np.testing.assert_allclose(y[1], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(y[2], [1 / 3, 1 / 3, 1 / 3])
    np.testing.assert_allclose(n, [2, 0, 3])
