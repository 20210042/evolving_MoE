import json

from evaluate import LUCA_SYSTEM_PROMPT, build_eval_prompt, evaluate_item
from train_sft import build_prompt_messages, resolve_expert_id, stringify_completion


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


def test_evaluate_can_use_luca_system_prompt_for_qasc():
    item = {
        "id": "q1",
        "dataset": "qasc",
        "domain": "qasc",
        "instruction": "What affects rain? (A) paint (B) local weather",
        "ground_truth": "B",
        "scoring_kind": "qasc",
    }
    prompt = build_eval_prompt(
        item,
        dataset_name="qasc",
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        prompt_system="luca",
    )
    assert prompt[0]["role"] == "system"
    assert prompt[0]["content"] == LUCA_SYSTEM_PROMPT
    assert "answer letter" in prompt[1]["content"]


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


def test_evaluate_can_use_luca_system_prompt_for_lbox():
    item = {
        "id": "l1",
        "dataset": "lbox",
        "domain": "lbox",
        "task_type": "casename",
        "instruction": "다음 사실관계에 해당하는 사건명을 정확히 한 줄로 답하라.",
        "ground_truth": "사기",
        "scoring_kind": "lbox",
    }
    prompt = build_eval_prompt(
        item,
        dataset_name="lbox",
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        prompt_system="luca",
    )
    assert prompt[0]["role"] == "system"
    assert prompt[0]["content"] == LUCA_SYSTEM_PROMPT
    assert "Korean legal classification" in prompt[1]["content"]


def test_evaluate_separates_math_baseline_and_category_prompts():
    item = {
        "id": "m1",
        "dataset": "numina_cot",
        "domain": "math",
        "category": "Geometry",
        "instruction": "Find x.",
        "ground_truth": "1",
    }
    baseline = build_eval_prompt(item, dataset_name="numina_cot", model_name="google/gemma")
    category = build_eval_prompt(
        item,
        dataset_name="numina_cot",
        model_name="google/gemma",
        prompt_system="category",
    )
    assert baseline[0]["content"] == "You are a helpful math assistant."
    assert "Euclidean, analytic, and projective geometry" in category[0]["content"]


def test_train_sft_expert_helpers_select_best_and_stringify_lists():
    mapping = {
        "a": {"train_pass_at_1": 10.0},
        "b": {"train_pass_at_1": 20.0},
    }
    assert resolve_expert_id(mapping, "best") == "b"
    assert resolve_expert_id(mapping, "a") == "a"
    assert stringify_completion(["형법 제298조", "형법 제299조"]) == "형법 제298조, 형법 제299조"
    assert stringify_completion({"answer": "B"}) == json.dumps({"answer": "B"}, ensure_ascii=False, sort_keys=True)


def test_train_sft_can_use_luca_system_prompt_for_qasc_lbox():
    qasc = {
        "id": "q1",
        "dataset": "qasc",
        "domain": "qasc",
        "instruction": "What affects rain? (A) paint (B) local weather",
        "ground_truth": "B",
    }
    lbox = {
        "id": "l1",
        "dataset": "lbox",
        "domain": "lbox",
        "instruction": "다음 사실관계에 해당하는 사건명을 정확히 한 줄로 답하라.",
        "ground_truth": "사기",
    }

    qasc_prompt = build_prompt_messages(
        qasc,
        "qasc",
        "meta-llama/Llama-3.1-8B-Instruct",
        prompt_system="luca",
    )
    lbox_prompt = build_prompt_messages(
        lbox,
        "lbox",
        "meta-llama/Llama-3.1-8B-Instruct",
        prompt_system="luca",
    )

    assert qasc_prompt[0]["content"] == LUCA_SYSTEM_PROMPT
    assert "answer letter" in qasc_prompt[1]["content"]
    assert lbox_prompt[0]["content"] == LUCA_SYSTEM_PROMPT
    assert "Korean legal classification" in lbox_prompt[1]["content"]
