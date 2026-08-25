import importlib.util
import json
from pathlib import Path


_p = Path(__file__).parents[1] / "scripts" / "router" / "aggregate_acc_sft_oracle.py"
_s = importlib.util.spec_from_file_location("aggregate_acc_sft_oracle", _p)
_m = importlib.util.module_from_spec(_s)
_s.loader.exec_module(_m)


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_aggregate_oracle_union(tmp_path):
    test = tmp_path / "test.jsonl"
    parts = tmp_path / "parts"
    parts.mkdir()
    _write(test, [{"id": "x"}, {"id": "y"}, {"id": "z"}])
    _write(parts / "a.jsonl", [
        {"id": "x", "prediction": "ax", "pass_score": 1},
        {"id": "y", "prediction": "ay", "pass_score": 0},
        {"id": "z", "prediction": "az", "pass_score": 0},
    ])
    _write(parts / "b.jsonl", [
        {"id": "x", "prediction": "bx", "pass_score": 0},
        {"id": "y", "prediction": "by", "pass_score": 1},
        {"id": "z", "prediction": "bz", "pass_score": 0},
    ])
    rows, summary = _m.aggregate(test, parts, ["a", "b"])
    assert summary["oracle_union_solved"] == 2
    assert summary["oracle_union_pass_at_1"] == 2 / 3
    assert summary["unique_solve_by_expert"] == {"a": 1, "b": 1}
    assert [r["oracle_pass"] for r in rows] == [1, 1, 0]
