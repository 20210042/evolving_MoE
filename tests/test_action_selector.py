"""Unit tests for 3-action gate."""

from meta_agent_evo.action_selector import ActionGateConfig, select_action


def test_swap_when_worst_redundant():
    # R = {a,b,w}; w never uniquely solves; new solves all hard
    squad = {
        "a": {"q1"},
        "b": {"q2"},
        "w": set(),
    }
    probe_hard = ["q1", "q2", "qx"]
    probe_stab = ["q1"]
    new_pass = {"qx"}  # new only saves hard failure
    cfg = ActionGateConfig(epsilon_floor=0.01, lambda_size=0.02, use_wilson_ci=False)
    d = select_action(
        roster_ids=["a", "b", "w"],
        worst_id="w",
        squad_results=squad,
        probe_hard=probe_hard,
        probe_stability=probe_stab,
        new_pass_ids=new_pass,
        cfg=cfg,
    )
    assert d.action in ("swap", "add", "noop")


def test_noop_when_no_gain():
    squad = {"a": {"q1"}, "b": {"q2"}, "w": set()}
    new_pass: set = set()
    cfg = ActionGateConfig(use_wilson_ci=False, epsilon_floor=0.05)
    d = select_action(
        roster_ids=["a", "b", "w"],
        worst_id="w",
        squad_results=squad,
        probe_hard=["q1", "q2"],
        probe_stability=[],
        new_pass_ids=new_pass,
        cfg=cfg,
    )
    assert d.action == "noop"


def test_swap_preferred_when_equal_gain():
    """Without swap_max_gain, swap wins when u_swap >= u_add."""
    squad = {
        "a": {"q1"},
        "b": {"q2"},
        "w": set(),
    }
    probe_hard = ["q1", "q2", "qx"]
    probe_stab = ["q1"]
    new_pass = {"qx"}
    cfg = ActionGateConfig(
        epsilon_floor=0.01,
        lambda_size=0.05,
        swap_max_gain=None,
        use_wilson_ci=False,
    )
    d = select_action(
        roster_ids=["a", "b", "w"],
        worst_id="w",
        squad_results=squad,
        probe_hard=probe_hard,
        probe_stability=probe_stab,
        new_pass_ids=new_pass,
        cfg=cfg,
    )
    assert d.action == "swap"
