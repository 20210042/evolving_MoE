import random
from war import pick_worst_agent
from orchestrator import GMEvolutionOrchestrator
from unittest.mock import MagicMock
import json

def test_pick_worst_agent_blocks_active_lives():
    """
    Verify that pick_worst_agent only targets agents with lives <= 0.
    """
    roster = [
        {"id": "A", "average_war": 0.0, "active_steps": 5, "lives": 3},
        {"id": "B", "average_war": 0.0, "active_steps": 5, "lives": 1},
        {"id": "C", "average_war": 0.0, "active_steps": 5, "lives": 0},
    ]
    war_scores = {"A": 0, "B": 0, "C": 0}
    rng = random.Random(42)
    
    # Only C has lives <= 0, so it must be nominated
    worst = pick_worst_agent(war_scores, roster, tiebreak="random", rng=rng)
    assert worst == "C"

def test_pick_worst_agent_returns_none_when_all_have_lives():
    """
    Verify that pick_worst_agent returns None if all active agents have lives > 0.
    """
    roster = [
        {"id": "A", "average_war": 0.0, "active_steps": 5, "lives": 2},
        {"id": "B", "average_war": 0.0, "active_steps": 5, "lives": 1},
    ]
    war_scores = {"A": 0, "B": 0}
    rng = random.Random(42)
    
    # None of the active agents have lives <= 0
    worst = pick_worst_agent(war_scores, roster, tiebreak="random", rng=rng)
    assert worst is None

def test_orchestrator_lives_update_and_recharge(tmp_path):
    """
    Verify orchestrator initial lives configuration, decrement on WAR=0, and recharge on WAR>0.
    """
    # 1. Setup roster file
    roster_file = tmp_path / "roster.json"
    initial_roster = [
        {"id": "agent_1", "name": "Agent 1"},
        {"id": "agent_2", "name": "Agent 2"},
    ]
    with open(roster_file, "w") as f:
        json.dump(initial_roster, f)
        
    # 2. Instantiate Orchestrator with max_lives=3
    mock_agent = MagicMock()
    orchestrator = GMEvolutionOrchestrator(
        agent=mock_agent,
        roster_path=str(roster_file),
        results_dir=str(tmp_path / "results"),
        run_id="test_lives_run",
        max_lives=3
    )
    
    # Verify both agents get max_lives by default
    assert orchestrator.roster[0]["lives"] == 3
    assert orchestrator.roster[1]["lives"] == 3
    
    # 3. Simulate Epoch 1: agent_1 has WAR=0 (loses a life), agent_2 has WAR=1 (recharges/retains 3 lives)
    squad_res = {"agent_1": set(), "agent_2": {"prob1"}}
    hard_errs = {}
    orchestrator.run_batch = MagicMock(return_value=(squad_res, hard_errs))
    
    from action_selector import ActionDecision
    import orchestrator
    orchestrator.scout_new_persona = MagicMock(return_value={"system_prompt": "new prompt", "persona_name": "new"})
    orchestrator.select_action = MagicMock(return_value=ActionDecision("noop", {}, 0.0, 0.0))
    orchestrator._run_candidate_on_item = MagicMock(return_value="code")
    orchestrator._score = MagicMock(return_value=1.0)
    
    # Run epoch 1
    orchestrator.run_epoch([{"id": "prob1"}])
    
    # agent_1 (WAR = 0) -> lives = 2
    # agent_2 (WAR = 1) -> lives = 3
    assert orchestrator.roster[0]["lives"] == 2
    assert orchestrator.roster[1]["lives"] == 3
    
    # 4. Simulate Epoch 2: agent_1 has WAR=1 (recharges to 3), agent_2 has WAR=0 (loses a life to 2)
    squad_res = {"agent_1": {"prob1"}, "agent_2": set()}
    orchestrator.run_batch = MagicMock(return_value=(squad_res, hard_errs))
    
    orchestrator.run_epoch([{"id": "prob1"}])
    
    # agent_1 (WAR = 1) -> recharges to 3
    # agent_2 (WAR = 0) -> lives = 2
    assert orchestrator.roster[0]["lives"] == 3
    assert orchestrator.roster[1]["lives"] == 2
