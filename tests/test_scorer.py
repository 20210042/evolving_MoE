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


def test_score_qasc_letter_and_option_text():
    item = {
        "id": "q1",
        "dataset": "qasc",
        "scoring_kind": "qasc",
        "instruction": "What affects rain? (A) Moon phase (B) local weather conditions (C) paint (D) music",
        "ground_truth": "B",
        "domain": "qasc",
    }
    assert score_one(item, "The answer is (B) local weather conditions") == 100.0
    assert score_one(item, "Final answer: C") == 0.0
    assert score_one(item, "local weather conditions") == 100.0


def test_score_lbox_casename_em():
    item = {
        "id": "l1",
        "dataset": "lbox",
        "scoring_kind": "lbox",
        "task_type": "casename",
        "ground_truth": "감염병의예방및관리에관한법률위반",
        "domain": "lbox",
    }
    assert score_one(item, "죄명: 감염병의 예방 및 관리에 관한 법률 위반") == 100.0
    assert score_one(item, "절도") == 0.0


def test_score_lbox_statute_set_em():
    item = {
        "id": "l2",
        "dataset": "lbox",
        "scoring_kind": "lbox",
        "task_type": "statute",
        "ground_truth": ["형법 제298조", "형법 제299조"],
        "domain": "lbox",
    }
    assert score_one(item, "형법 제299조, 형법 제298조") == 100.0
    assert score_one(item, "형법 제298조") == 0.0
