from pathlib import Path

from roster import assign_candidate_id, ensure_roster


def test_default_roster(tmp_path):
    p = tmp_path / "r.json"
    r = ensure_roster(str(p))
    assert len(r) == 5
    assert r[0]["id"] == "array"


def test_assign_id_unique():
    roster = [{"id": "a"}]
    nid = assign_candidate_id(roster)
    assert nid.startswith("c_")


def test_normalize_persona_fields_initializes_metrics():
    from meta_agent_evo.roster import normalize_persona_fields
    persona = {"persona_name": "Test Critic"}
    normalized = normalize_persona_fields(persona, "c_test")
    assert normalized["id"] == "c_test"
    assert normalized["total_war"] == 0
    assert normalized["active_steps"] == 0
    assert normalized["average_war"] == 0.0


def test_pick_worst_agent_cumulative_metrics():
    from meta_agent_evo.war import pick_worst_agent
    import random
    
    # Roster with cumulative metrics
    # Candidate A: avg_war = 0.0, active_steps = 1
    # Candidate B: avg_war = 0.0, active_steps = 5  (Proven useless, should be evicted first)
    # Candidate C: avg_war = 0.5, active_steps = 2
    roster = [
        {"id": "A", "average_war": 0.0, "active_steps": 1, "lives": 0},
        {"id": "B", "average_war": 0.0, "active_steps": 5, "lives": 0},
        {"id": "C", "average_war": 0.5, "active_steps": 2, "lives": 3},
    ]
    
    war_scores = {"A": 0, "B": 0, "C": 1}
    rng = random.Random(42)
    worst = pick_worst_agent(war_scores, roster, tiebreak="random", rng=rng)
    assert worst == "B"


def test_orchestrator_cumulative_metrics(tmp_path):
    from unittest.mock import MagicMock
    from meta_agent_evo.orchestrator import GMEvolutionOrchestrator
    import json
    
    # 1. Prepare roster path and data
    roster_file = tmp_path / "roster.json"
    initial_roster = [
        {"id": "array", "name": "Array Specialist"},
        {"id": "string", "name": "String Specialist"},
    ]
    with open(roster_file, "w") as f:
        json.dump(initial_roster, f)
        
    # 2. Instantiate Orchestrator with mock Agent
    mock_agent = MagicMock()
    orchestrator = GMEvolutionOrchestrator(
        agent=mock_agent,
        roster_path=str(roster_file),
        results_dir=str(tmp_path / "results"),
        run_id="test_run",
    )
    
    # Check initialization default fields
    assert orchestrator.roster[0]["total_war"] == 0
    assert orchestrator.roster[0]["active_steps"] == 0
    assert orchestrator.roster[0]["average_war"] == 0.0
    
    # 3. Mock run_batch to return custom outcomes
    squad_res = {"array": {"p1"}, "string": set()}
    hard_errs = {"p2": "sample error"}
    orchestrator.run_batch = MagicMock(return_value=(squad_res, hard_errs))
    
    # Mock scout_new_persona and select_action to do "noop" so roster size remains 2
    from meta_agent_evo.action_selector import ActionDecision
    import meta_agent_evo.orchestrator
    
    meta_agent_evo.orchestrator.scout_new_persona = MagicMock(return_value={"system_prompt": "new prompt", "persona_name": "new"})
    meta_agent_evo.orchestrator.select_action = MagicMock(return_value=ActionDecision("noop", {}, 0.0, 0.0))
    orchestrator._run_candidate_on_item = MagicMock(return_value="code")
    orchestrator._score = MagicMock(return_value=1.0)
    
    # Run first epoch
    batch_data = [{"id": "p1"}, {"id": "p2"}]
    orchestrator.run_epoch(batch_data)
    
    # Verify cumulative updates for Epoch 1
    assert orchestrator.roster[0]["id"] == "array"
    assert orchestrator.roster[0]["total_war"] == 1
    assert orchestrator.roster[0]["active_steps"] == 1
    assert orchestrator.roster[0]["average_war"] == 1.0
    
    assert orchestrator.roster[1]["id"] == "string"
    assert orchestrator.roster[1]["total_war"] == 0
    assert orchestrator.roster[1]["active_steps"] == 1
    assert orchestrator.roster[1]["average_war"] == 0.0
    
    # Check that it saved the updated roster to disk
    with open(roster_file, "r") as f:
        saved_roster = json.load(f)
    assert saved_roster[0]["total_war"] == 1
    assert saved_roster[0]["active_steps"] == 1
    assert saved_roster[0]["average_war"] == 1.0
