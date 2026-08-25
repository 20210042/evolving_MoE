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


def test_score_acc_strips_markdown_fence_before_execution(monkeypatch):
    captured = {}

    def fake_run(
        self,
        problem,
        candidate_code,
        language="python",
        solution_id="candidate",
    ):
        captured["candidate_code"] = candidate_code
        return {"passed": True}

    monkeypatch.setattr(
        "evaluation.acc_exec.ExecutionInterface.run",
        fake_run,
    )

    item = {
        "id": "acc_fenced",
        "dataset": "acc",
        "scoring_kind": "acc",
        "instruction": "Write a Python solution.",
        "eval_spec": {"eval_mode": "stdin_stdout"},
        "test_cases": [],
        "domain": "coding",
    }

    score = score_one(item, "```python\nprint('hello')\n```")

    assert score == 100.0
    assert captured["candidate_code"] == "print('hello')"


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


def test_score_qasc_enumerated_options_do_not_override_answer():
    """보기를 전부 나열한 뒤 답을 말하는 장황한 출력도 정답으로 잡혀야 한다.

    회귀: 세 추출 패턴의 매치를 한 리스트에 모아 마지막을 취하던 구현에서는
    "(A)...(D)" 나열의 마지막 글자가 명시된 답을 덮어써서, 장황한 에이전트가
    구조적으로 오답 처리됐다.
    """
    item = {
        "id": "q2",
        "dataset": "qasc",
        "scoring_kind": "qasc",
        "instruction": "What affects rain? (A) Moon phase (B) local weather conditions (C) paint (D) music",
        "ground_truth": "B",
        "domain": "qasc",
    }
    verbose = (
        "Let's evaluate each option:\n"
        "(A) Moon phase - no measurable effect.\n"
        "(B) local weather conditions - this drives precipitation.\n"
        "(C) paint - irrelevant.\n"
        "(D) music - irrelevant.\n\n"
        "Final answer: B"
    )
    assert score_one(item, verbose) == 100.0
    # 답을 마지막 줄에 글자로만 두는 형태도 동일하게 동작해야 한다.
    assert score_one(item, verbose.replace("Final answer: B", "B")) == 100.0
    # 나열만 있고 답 표시가 없으면 마지막 보기로 떨어지는 기존 동작은 유지.
    assert score_one(item, "(A) Moon phase (B) local weather conditions") == 100.0


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
