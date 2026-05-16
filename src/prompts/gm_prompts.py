from string import Template

# 1. Manager Prompt (For Phase 2: Zero-Shot Routing)
# The manager receives the problem description and the roster JSON, then returns the ID of the chosen Critic.
MANAGER_PROMPT = Template("""You are the General Manager (Router) of a software engineering team.
Your task is to analyze the following programming problem and select exactly one expert from your current roster who is best suited to solve it.

=== Current Roster Scouting Report ===
$scouting_report

=== Problem Description ===
$problem_description

Based on the problem description, select the single best critic to handle this task.
You MUST output your selection in valid JSON format exactly as follows:
{
    "selected_critic_id": "the_id_of_the_chosen_critic"
}
Do not output any other text or explanation. Only the JSON object.
""")

# 2. Meta Agent Prompt (The Strategic Scouter)
META_AGENT_PROMPT = Template("""You are the Head Scouter for an AI Competitive Programming Team. 

Your mission is to analyze recent failures and define a specialist persona to fill the technical gaps identified in those failures.

=== FAILURE DIAGNOSIS (Hard Errors) ===
$hard_errors

=== CURRENT ROSTER (Active Personnel) ===
$current_roster

--- SCOUTING ORDERS ---
1. DIAGNOSE: Analyze the "Hard Errors" deeply. Determine if failures are due to algorithmic complexity (TLE), edge case handling, or input/output formatting.
2. GAP ANALYSIS: Identify why the current roster failed. Determine what specific technical expertise or focus is missing.

Based on this, define a new specialist persona in the following JSON format:
{
    "persona_name": "A clear, descriptive name (e.g., DP_Optimization_Specialist)",
    "system_prompt": "A detailed system prompt defining the persona's technical expertise and focus for solving the identified failures.",
    "strengths": "Short description of specific expertise and problem types to solve",
    "custom_refine_prompt_template": "A precise instruction (1-2 sentences) dictating exactly what to check for in the candidate code."
}
""")
