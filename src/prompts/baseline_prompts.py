# Baseline Prompts (Revised: Few-Shot Generator, Zero-Shot Critic/Revision)

# --- CODING PROMPTS (MBPP Style) ---

CODING_GEN_SYSTEM = "You are an expert programmer."
CODING_GEN_USER = """Write a Python function to solve the following problem.

### Task
Problem:
{instruction}

```python
[CODE]
```
"""

# --- LIVECODEBENCH PROMPTS (Stock / No Persona) ---
LCB_GEN_SYSTEM = "You are a helpful assistant."
LCB_GEN_USER = """Solve the following competitive programming problem using Python 3.
Read from standard input (stdin) and write to standard output (stdout).

### Task
Problem:
{instruction}

Required format:
```python
[CODE]
```
"""


# DS-1000: Specialized for Code Completion tasks
DS1000_GEN_SYSTEM = "You are an expert Data Science code completion assistant. Output only the code that completes the provided context."
# Note: We provide the raw instruction (which is the DS-1000 prompt) directly.
DS1000_GEN_USER = """{instruction}"""

# Critic: Zero-Shot, Textual Feedback Only
CODING_CRITIC_SYSTEM = "You are a helpful assistant."
CODING_CRITIC_USER = """Review the following code.

Problem:
{instruction}

Code:
{code}

1. Identify any syntax errors or bugs.
2. Check if it solves the problem correctly.
3. Assess the efficiency.

Output your review in the following format:
Feedback: [Your detailed feedback]
"""

# Revision: Zero-Shot
CODING_REVISION_SYSTEM = "You are an expert programmer."
CODING_REVISION_USER = """Refine the code based on the feedback.

Problem:
{instruction}

Previous Code:
{code}

Feedback:
{feedback}

Return the improved code in the following format:
```python
[CODE]
```
"""

# --- MATH PROMPTS (GSM8K Style) ---

MATH_GEN_SYSTEM = "You are a helpful math assistant."
MATH_GEN_SYSTEM_BY_DOMAIN = {
    "algebra": (
        "You are an expert in foundational algebra and trigonometry. Focus on solving "
        "problems involving sequences and series, trigonometric identities, and linear "
        "algebraic systems."
    ),
    "geometry": (
        "You are an expert in Euclidean, analytic, and projective geometry. You "
        "specialize in coordinate geometry of conic sections and 3D spatial reasoning."
    ),
    "number_theory": (
        "You specialize in combinatorial proofs, elementary number theory, and formal "
        "logic. Focus on set theory, divisibility, pigeonhole principle, and rigorous "
        "deductive reasoning."
    ),
    "combinatorics": (
        "You specialize in combinatorial proofs, elementary number theory, and formal "
        "logic. Focus on set theory, divisibility, pigeonhole principle, and rigorous "
        "deductive reasoning."
    ),
    "calculus": (
        "You are an expert in mathematical analysis, specializing in differential and "
        "integral calculus. Focus on rigor in determining monotonicity, extrema, and "
        "the behavior of transcendental functions."
    ),
}
MATH_GEN_USER = """Solve the following math problem. Show your work step-by-step, and put your final answer in \\boxed{{}}.

Question:
{instruction}
"""

# --- SCIENCE MULTIPLE-CHOICE PROMPTS (QASC) ---

QASC_GEN_SYSTEM = "You are a careful science multiple-choice solver."
QASC_GEN_USER = """Solve the following science multiple-choice question.
Output only the final answer letter, one of A, B, C, D, E, F, G, or H.

Question:
{instruction}
"""
QASC_CRITIC_SYSTEM = "You are a careful science multiple-choice reviewer."
QASC_CRITIC_USER = """Review the proposed answer to this science multiple-choice question.

Question:
{instruction}

Answer:
{solution}

Check whether the final letter is correct. If it is already correct, say no change is needed.
Feedback: [Your concise feedback]
"""
QASC_REVISION_SYSTEM = "You are a careful science multiple-choice solver."
QASC_REVISION_USER = """Revise the answer based on the feedback.

Question:
{instruction}

Previous Answer:
{solution}

Feedback:
{feedback}

Output only the final answer letter.
"""

# --- KOREAN LEGAL EM PROMPTS (LBox Open) ---

LBOX_GEN_SYSTEM = "You are a precise Korean legal classification assistant."
LBOX_GEN_USER = """Answer the following Korean legal classification task exactly in the requested format.
Do not add explanations.

Task:
{instruction}
"""
LBOX_CRITIC_SYSTEM = "You are a precise Korean legal classification reviewer."
LBOX_CRITIC_USER = """Review the proposed answer to this Korean legal classification task.

Task:
{instruction}

Answer:
{solution}

Check exactness against the requested output format. If it is already exact, say no change is needed.
Feedback: [Your concise feedback]
"""
LBOX_REVISION_SYSTEM = "You are a precise Korean legal classification assistant."
LBOX_REVISION_USER = """Revise the answer based on the feedback.

Task:
{instruction}

Previous Answer:
{solution}

Feedback:
{feedback}

Output only the corrected final answer.
"""

# Super-NaturalInstructions: instruction 안에 이미 태스크 Definition + Input이 들어 있다.
# 여기서 태스크를 다시 설명하지 않는다 — 우리가 덧붙이는 건 "설명 없이 출력만" 계약뿐이다.
# ROUGE-L은 여분의 텍스트를 그대로 페널티로 먹으므로 이 계약이 곧 점수다.
SNI_GEN_SYSTEM = "You follow task instructions exactly and output only the requested answer."
# ⚠️ user 턴은 **답변공간 한 줄(answer_line) + 입력**뿐이다. 태스크 정의(Definition)를 여기
# 넣으면 조작과 출력형식이 못박혀 system의 정체성이 개입할 자리가 사라진다 —
# 프로브 job 229352의 null이 그것이었다(docs/REFLECTION_sni_probe.md).
# answer_line은 export가 데이터에서 기계적으로 만든다(scripts/build_sni_export.py).
SNI_GEN_USER = """{answer_line}

{instruction}
"""
SNI_CRITIC_SYSTEM = "You review whether a response follows the task instruction exactly."
SNI_CRITIC_USER = """Review the proposed response to this task.

{instruction}

Response:
{solution}

Check whether it follows the task definition and contains only the requested answer.
If it is already correct, say no change is needed.
Feedback: [Your concise feedback]
"""
SNI_REVISION_SYSTEM = "You follow task instructions exactly and output only the requested answer."
SNI_REVISION_USER = """Revise the response based on the feedback.

{instruction}

Previous Response:
{solution}

Feedback:
{feedback}

Output only the corrected answer itself.
"""

# Critic: Zero-Shot, Textual Feedback Only
MATH_CRITIC_SYSTEM = "You are a helpful assistant."
MATH_CRITIC_USER = """Review the following solution.

Question:
{instruction}

Solution:
{solution}

1. Check for logical errors.
2. Verify calculations.

Output your review in the following format:
Feedback: [Your detailed feedback]
"""

# Revision: Zero-Shot
MATH_REVISION_SYSTEM = "You are a helpful math assistant."
MATH_REVISION_USER = """Refine the solution based on the feedback.

Question:
{instruction}

Previous Solution:
{solution}

Feedback:
{feedback}

Provide the corrected solution.
Answer format:
Final Answer: [ANSWER]
"""
