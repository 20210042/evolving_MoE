from data.sni import build_sni_prompt, select_sni_target


def test_build_sni_prompt_uses_exact_requested_parts_and_only_two_examples():
    item = {
        "definition": "Define the task.",
        "positive_examples": [
            {"input": "first input", "output": "first output", "explanation": "omit me"},
            {"input": "second input", "output": "second output"},
            {"input": "third input", "output": "third output"},
        ],
        "instruction": "target instruction",
    }

    prompt = build_sni_prompt(item)

    assert prompt.startswith("Definition:\nDefine the task.")
    assert "Positive Example 1:\nInput:\nfirst input\n\nOutput:\nfirst output" in prompt
    assert "Positive Example 2:\nInput:\nsecond input\n\nOutput:\nsecond output" in prompt
    assert "third input" not in prompt
    assert "omit me" not in prompt
    assert prompt.endswith("Instruction:\ntarget instruction")


def test_select_sni_target_is_one_ground_truth_and_reproducible():
    item = {"id": "row-1", "ground_truth": ["a", "b", "c"]}

    first = select_sni_target(item, seed=42)

    assert first in item["ground_truth"]
    assert first == select_sni_target(item, seed=42)
