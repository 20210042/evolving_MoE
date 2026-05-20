from evaluation.scorer import score_one


def test_score_mbpp_zero_on_garbage():
    item = {
        "id": "t1",
        "dataset": "mbpp",
        "scoring_kind": "asserts",
        "instruction": "dummy",
        "ground_truth": "def f():\n    return 1",
        "test_list": ["assert f()==1"],
        "domain": "coding",
    }
    s = score_one(item, "not python {{{", code_timeout=1.0)
    assert s == 0.0
