# Baseline Prompts (Revised: Few-Shot Generator, Zero-Shot Critic/Revision)

# --- CODING PROMPTS (MBPP Style) ---

CODING_GEN_SYSTEM = "You are an expert programmer."
CODING_GEN_USER = """Write a Python function to solve the following problem.

### Example
Problem: Write a function to check if a number is even.
Code:
```python
def is_even(n):
    return n % 2 == 0
```
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

# --- MATH PROMPTS ---

MATH_GEN_SYSTEM = "You are a helpful math assistant."
MATH_GEN_USER = """Solve the following math problem. Show your work step-by-step.
Write your final answer after "Final Answer:".
Use LaTeX format for fractions and expressions (e.g. \\frac{{a}}{{b}}, \\sqrt{{n}}).

### Example
Question: If cos θ = 3/5 and θ is in the first quadrant, find sin θ.
Final Answer: \\frac{{4}}{{5}}

### Task
Question:
{instruction}
Final Answer: """

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
