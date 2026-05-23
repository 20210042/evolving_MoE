"""Unit tests for Phase 1 / Phase 2 independent Action Gate with Exponential Penalty."""

from meta_agent_evo.action_selector import ActionGateConfig, select_action


def test_swap_when_worst_redundant():
    # R = {a,b,w}; N = 3
    # New candidate solves the 1 hard error (gain = 1/10 = 10%)
    squad = {
        "a": {"q1"},
        "b": {"q2"},
        "w": set(),
    }
    hard_errors = ["qx"]
    new_pass = {"qx"}
    cfg = ActionGateConfig(lambda_size=0.10)
    
    d = select_action(
        roster_ids=["a", "b", "w"],
        worst_id="w",
        squad_results=squad,
        hard_errors=hard_errors,
        new_pass_ids=new_pass,
        batch_size=10,
        cfg=cfg,
    )
    # N = 3, lambda = 0.10
    # factor = 0.5 * (exp(0.10) - 1.0) = 0.0526
    # lambda_add = exp(3 * 0.10) * 0.0526 = 1.3498 * 0.0526 = 0.071
    # lambda_del = exp(2 * 0.10) * 0.0526 = 1.2214 * 0.0526 = 0.064
    # gh_add = 1/10 = 0.10
    # u_add = 0.10 - 0.071 = 0.029 > 0.0 (add = True)
    # mcl = 0
    # u_delete = 0.064 - 0 = 0.064 > 0.0 (delete = True)
    # Together: swap
    assert d.action == "swap"


def test_noop_when_no_gain():
    # R = {a,b,w}; N = 3
    squad = {"a": {"q1"}, "b": {"q2"}, "w": {"q3"}}
    new_pass: set = set()
    cfg = ActionGateConfig(lambda_size=0.10)
    
    d = select_action(
        roster_ids=["a", "b", "w"],
        worst_id="w",
        squad_results=squad,
        hard_errors=["qx"],
        new_pass_ids=new_pass,
        batch_size=10,
        cfg=cfg,
    )
    # lambda_add = 0.071
    # lambda_del = 0.064
    # gh_add = 0.0
    # u_add = -0.071 (stay)
    # w solves q3 uniquely, mcl = 1/10 = 0.10
    # u_delete = 0.064 - 0.10 = -0.036 <= 0.0 (stay)
    # Together: noop
    assert d.action == "noop"


def test_delete_only():
    # R = {a,w}; N = 2
    squad = {
        "a": {"q1", "q2"},
        "w": {"q3"},  # w uniquely solves q3
    }
    hard_errors = ["qx"]
    new_pass: set = set()
    cfg = ActionGateConfig(lambda_size=0.20)
    
    d = select_action(
        roster_ids=["a", "w"],
        worst_id="w",
        squad_results=squad,
        hard_errors=hard_errors,
        new_pass_ids=new_pass,
        batch_size=10,
        cfg=cfg,
    )
    # N = 2, lambda = 0.20
    # factor = 0.5 * (exp(0.20) - 1.0) = 0.1107
    # lambda_add = exp(2 * 0.20) * 0.1107 = 1.4918 * 0.1107 = 0.165
    # lambda_del = exp(1 * 0.20) * 0.1107 = 1.2214 * 0.1107 = 0.135
    # gh_add = 0.0; u_add = -0.165 (stay)
    # mcl = 1/10 = 0.10
    # u_delete = 0.135 - 0.10 = 0.035 > 0.0 (delete = True)
    # Together: delete
    assert d.action == "delete"
