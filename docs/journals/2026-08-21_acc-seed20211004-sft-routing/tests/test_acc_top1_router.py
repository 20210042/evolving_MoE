import importlib.util
from pathlib import Path

import numpy as np
import torch


_p = Path(__file__).parents[1] / "scripts" / "router" / "train_acc_top1_router.py"
_s = importlib.util.spec_from_file_location("train_acc_top1_router", _p)
_m = importlib.util.module_from_spec(_s)
_s.loader.exec_module(_m)


def test_solve_targets_preserve_multi_hot_labels():
    rows = [
        {"per_expert": {"a": 1, "b": 0, "c": 1}},
        {"per_expert": {"a": 0, "b": 0, "c": 0}},
    ]
    y, n = _m.solve_targets(rows, ["a", "b", "c"])
    np.testing.assert_array_equal(y, [[1, 0, 1], [0, 0, 0]])
    np.testing.assert_array_equal(n, [2, 0])


def test_set_mass_loss_rewards_probability_on_any_solver():
    targets = torch.tensor([[1.0, 0.0, 1.0]])
    good = _m.set_mass_loss(torch.tensor([[5.0, -5.0, -5.0]]), targets)
    also_good = _m.set_mass_loss(torch.tensor([[-5.0, -5.0, 5.0]]), targets)
    bad = _m.set_mass_loss(torch.tensor([[-5.0, 5.0, -5.0]]), targets)
    assert good < bad
    assert also_good < bad


def test_all_pass_has_zero_set_loss():
    logits = torch.tensor([[2.0, -1.0, 0.5]])
    targets = torch.ones_like(logits)
    assert torch.allclose(_m.set_mass_loss(logits, targets), torch.tensor(0.0), atol=1e-6)
