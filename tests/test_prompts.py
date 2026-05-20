"""Unit tests for prompt family routing (no vLLM)."""

from meta_agent_evo.prompts import coding


def test_get_prompt_family_qwen():
    assert coding.get_prompt_family("Qwen/Qwen3-Coder-30B-A3B-Instruct") == "qwen3"


def test_get_prompt_family_llama():
    assert coding.get_prompt_family("meta-llama/Meta-Llama-3.1-8B-Instruct") == "llama31"


def test_get_prompt_family_generic():
    assert coding.get_prompt_family("vendor/SomeModel") == "generic"


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
