import re


def extract_code_block(text: str) -> str:
    """
    Robustly extract code from model output, handling markdown,
    custom tags ([BEGIN], <code>), and common suffixes.
    """
    if not text:
        return ""

    code = text
    if "[BEGIN]" in code and "[DONE]" in code:
        code = code.split("[BEGIN]")[1].split("[DONE]")[0]

    if "```python" in code:
        code = code.split("```python")[1].split("```")[0]
    elif "```" in code:
        parts = code.split("```")
        if len(parts) >= 3:
            code = parts[1]
        elif len(parts) == 2:
            p0, p1 = parts[0].strip(), parts[1].strip()
            if p0 and not p1:
                code = p0
            elif p1 and not p0:
                code = p1
            else:
                code = parts[0] if len(parts[0]) > len(parts[1]) else parts[1]

    if "<code>" in code:
        code = code.split("<code>")[-1]
    if "</code>" in code:
        code = code.split("</code>")[0]

    return code.strip()


def check_stop_condition(feedback: str) -> bool:
    feedback_lower = feedback.lower()

    score_match = re.search(r"(?:Score|Grade|Rating):\s*(\d+)", feedback, re.IGNORECASE)
    if score_match:
        try:
            score = int(score_match.group(1))
            if score <= 10:
                return score >= 9
            if score <= 100:
                return score >= 90
        except ValueError:
            pass

    strong_signals = [
        "perfect",
        "no issues",
        "no bugs",
        "excellent",
        "flawless",
        "no errors",
        "completely correct",
        "works correctly",
    ]

    if "incorrect" in feedback_lower or "not correct" in feedback_lower or "bug" in feedback_lower:
        return False

    matched = sum(1 for sig in strong_signals if sig in feedback_lower)
    return matched >= 2
