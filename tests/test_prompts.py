"""Unit tests for prompt family routing (no vLLM)."""

from prompts import coding


def test_get_prompt_family_qwen():
    assert coding.get_prompt_family("Qwen/Qwen3-Coder-30B-A3B-Instruct") == "qwen3"


def test_get_prompt_family_llama():
    assert coding.get_prompt_family("meta-llama/Meta-Llama-3.1-8B-Instruct") == "llama31"


def test_get_prompt_family_generic():
    assert coding.get_prompt_family("vendor/SomeModel") == "generic"


def test_get_prompt_family_gemma():
    assert coding.get_prompt_family("google/gemma-4-31B-it") == "llama31"


def test_qwen_baseline_string():
    msg = coding.build_baseline_prompt("x", dataset="mbpp", model_name="Qwen3-x")
    assert isinstance(msg, str)


def test_llama_baseline_messages():
    msgs = coding.build_baseline_prompt(
        "x", dataset="mbpp", model_name="meta-llama/Meta-Llama-3.1-8B-Instruct"
    )
    assert isinstance(msgs, list)
    assert msgs[0]["role"] == "system"
    assert "```python" in msgs[0]["content"]


def test_qasc_prompt_is_letter_only_not_code():
    msgs = coding.build_baseline_prompt("Q? (A) x (B) y", dataset="qasc", model_name="google/gemma")
    assert isinstance(msgs, list)
    assert "answer letter" in msgs[1]["content"]
    assert "```python" not in msgs[0]["content"]
    assert "```python" not in msgs[1]["content"]


def test_lbox_prompt_is_legal_not_code():
    msgs = coding.build_expert_prompt(
        "다음 사실관계에 해당하는 죄명을 정확히 한 줄로 답하라.",
        "You specialize in Korean criminal law.",
        dataset="lbox",
        model_name="google/gemma",
        domain="lbox",
    )
    assert isinstance(msgs, list)
    assert "Korean legal classification" in msgs[1]["content"]
    assert "```python" not in msgs[0]["content"]
    assert "```python" not in msgs[1]["content"]
