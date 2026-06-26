"""Unit tests for prompt family routing (no vLLM)."""

from prompts import coding
from prompts import baseline_prompts
from prompts.math import build_generation_prompt, resolve_math_system_prompt


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


def test_math_prompt_uses_default_system_without_domain_metadata():
    msgs = build_generation_prompt("x")
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == baseline_prompts.MATH_GEN_SYSTEM


def test_math_prompt_uses_numina_category_specific_system():
    msgs = build_generation_prompt("x", metadata={"category": "Geometry"})
    assert msgs[0]["role"] == "system"
    assert "Euclidean, analytic, and projective geometry" in msgs[0]["content"]


def test_math_prompt_resolves_numina_number_theory_category():
    system_prompt = resolve_math_system_prompt({"category": "Number Theory"})
    assert "elementary number theory" in system_prompt


def test_math_prompt_resolves_numina_combinatorics_category():
    system_prompt = resolve_math_system_prompt({"category": "Combinatorics"})
    assert "combinatorial proofs" in system_prompt
