import re

def extract_code_block(text: str) -> str:
    """
    Robustly extract code from model output, handling markdown, 
    custom tags ([BEGIN], <code>), and common suffixes.
    """
    if not text:
        return ""
        
    code = text
    # 1. Handle custom [BEGIN]/[DONE] tags
    if "[BEGIN]" in code and "[DONE]" in code:
        code = code.split("[BEGIN]")[1].split("[DONE]")[0]
        
    # 2. Handle Markdown triple backticks
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0]
    elif "```" in code:
        parts = code.split("```")
        if len(parts) >= 3:
            # Standard block: text ```code``` text
            code = parts[1]
        elif len(parts) == 2:
            # Case 1: code ``` (Closing tag only) - common with assistant prefixes
            # Case 2: ``` code (Opening tag only)
            p0, p1 = parts[0].strip(), parts[1].strip()
            if p0 and not p1:
                code = p0
            elif p1 and not p0:
                code = p1
            else:
                code = parts[0] if len(parts[0]) > len(parts[1]) else parts[1]
    
    # 3. Handle <code> tags (common in DS-1000 prompts)
    if "<code>" in code:
        code = code.split("<code>")[-1] 
    if "</code>" in code:
        code = code.split("</code>")[0]
        
    return code.strip()

def check_stop_condition(feedback: str) -> bool:
    """
    Improved stop condition (Option C):
    - If explicit score exists: stop if score >= 9/10 or >= 90/100
    - If no score: require 2+ strong positive signals to stop
    """
    feedback_lower = feedback.lower()
    
    # Priority 1: Check for explicit Score/Grade
    score_match = re.search(r"(?:Score|Grade|Rating):\s*(\d+)", feedback, re.IGNORECASE)
    if score_match:
        try:
            score = int(score_match.group(1))
            if score <= 10:
                return score >= 9
            elif score <= 100:
                return score >= 90
        except ValueError:
            pass 

    # Priority 2: Require 2+ strong positive signals
    strong_signals = ["perfect", "no issues", "no bugs", "excellent", "flawless", 
                      "no errors", "completely correct", "works correctly"]
    
    # Negative signals
    if "incorrect" in feedback_lower or "not correct" in feedback_lower or "bug" in feedback_lower:
        return False
        
    matched = sum(1 for sig in strong_signals if sig in feedback_lower)
    if matched >= 2:
        return True

    return False
