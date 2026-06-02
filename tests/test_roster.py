from pathlib import Path

from roster import assign_candidate_id, ensure_roster


def test_default_roster(tmp_path):
    p = tmp_path / "r.json"
    r = ensure_roster(str(p))
    assert len(r) == 3
    assert r[0]["id"] == "senior_dev"


def test_assign_id_unique():
    roster = [{"id": "a"}]
    nid = assign_candidate_id(roster)
    assert nid.startswith("c_")
