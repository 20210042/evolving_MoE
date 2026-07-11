import json

from evaluate import build_eval_prompt, evaluate_item
from train_sft import resolve_expert_id, stringify_completion


def test_evaluate_builds_qasc_prompt_and_scores_letter():
    item = {
        "id": "q1",
        "dataset": "qasc",
        "domain": "qasc",
        "instruction": "What affects rain? (A) paint (B) local weather (C) music (D) rocks",
        "ground_truth": "B",
        "scoring_kind": "qasc",
    }
    prompt = build_eval_prompt(item, dataset_name="qasc", model_name="google/gemma")
    assert prompt[0]["role"] == "system"
    assert "science multiple-choice" in prompt[0]["content"]
    assert "answer letter" in prompt[1]["content"]
    assert evaluate_item(item, "Final answer: B", is_math_dataset=False)["pass_score"] == 100.0


def test_evaluate_builds_lbox_prompt_and_scores_statutes():
    item = {
        "id": "l1",
        "dataset": "lbox",
        "domain": "lbox",
        "task_type": "statute",
        "instruction": "다음 사실관계에 적용되는 법조문을 모두 나열하라.",
        "ground_truth": ["형법 제298조", "형법 제299조"],
        "scoring_kind": "lbox",
    }
    prompt = build_eval_prompt(item, dataset_name="lbox", model_name="google/gemma")
    assert prompt[0]["role"] == "system"
    assert "Korean legal classification" in prompt[0]["content"]
    assert evaluate_item(item, "형법 제299조, 형법 제298조", is_math_dataset=False)["pass_score"] == 100.0


def test_train_sft_expert_helpers_select_best_and_stringify_lists():
    mapping = {
        "a": {"train_pass_at_1": 10.0},
        "b": {"train_pass_at_1": 20.0},
    }
    assert resolve_expert_id(mapping, "best") == "b"
    assert resolve_expert_id(mapping, "a") == "a"
    assert stringify_completion(["형법 제298조", "형법 제299조"]) == "형법 제298조, 형법 제299조"
    assert stringify_completion({"answer": "B"}) == json.dumps({"answer": "B"}, ensure_ascii=False, sort_keys=True)
