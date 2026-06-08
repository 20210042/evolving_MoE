from string import Template

MANAGER_PROMPT = Template(
    """You are the Router of a software engineering team.
Assign the following problem to exactly one specialist from your roster.

=== Current Roster ===
$scouting_report

=== Problem ===
$problem_description

Pick the specialist most likely to solve this problem.
Output only valid JSON:
{
    "selected_expert_id": "the_id_of_the_chosen_expert"
}
"""
)

MANAGER_MATH_PROMPT = Template(
    """You are the Router of a mathematics team.
Assign the following problem to exactly one specialist from your roster.

=== Current Roster ===
$scouting_report

=== Problem ===
$problem_description

Pick the specialist most likely to solve this problem.
Output only valid JSON:
{
    "selected_expert_id": "the_id_of_the_chosen_expert"
}
"""
)

META_AGENT_PROMPT = Template(
    """You are the Head Scouter for an AI Competitive Programming Team.

Every agent on the current roster failed the following problems:

=== HARD ERRORS ===
$hard_errors

=== CURRENT ROSTER ===
$current_roster

--- YOUR TASK ---
Look at these unsolved problems. What expert is missing from this roster?

Define that expert yourself.

Rules:
1. NON-REDUNDANCY (CRITICAL): Must be genuinely different from every current roster member.
2. ATOMICITY (CRITICAL): One expert, one focused identity — not a combination of multiple.
3. persona_name must NOT contain the word 'and'.

Output in JSON. Keep system_prompt under 3 sentences:
{
    "persona_name": "...",
    "system_prompt": "...",
    "strengths": "..."
}
"""
)

META_AGENT_MATH_PROMPT = Template(
    """You are the Head Scouter for an AI Mathematics Team.

Every agent on the current roster failed the following problems:

=== HARD ERRORS ===
$hard_errors

=== CURRENT ROSTER ===
$current_roster

--- YOUR TASK ---
Look at these unsolved problems. What expert is missing from this roster?

Define that expert yourself.

Rules:
1. NON-REDUNDANCY (CRITICAL): Must be genuinely different from every current roster member.
2. ATOMICITY (CRITICAL): One expert, one focused identity — not a combination of multiple.
3. persona_name must NOT contain the word 'and'.

Output in JSON. Keep system_prompt under 3 sentences:
{
    "persona_name": "...",
    "system_prompt": "...",
    "strengths": "..."
}
"""
)

# Seed 20210009+: full minimal-intervention version.
# NON-REDUNDANCY and ATOMICITY rules removed — replaced by per-agent exclusive solve history.
# Scout sees what each agent actually solves alone; no text-level constraints.
META_AGENT_MATH_PROMPT_V2 = Template(
    """You are the Head Scouter for an AI Mathematics Team.

Every agent on the current roster failed the following problems:

=== HARD ERRORS ===
$hard_errors

=== CURRENT ROSTER ===
$current_roster

=== WHAT EACH AGENT HAS EXCLUSIVELY SOLVED (problems only they got right) ===
$exclusive_solves

--- YOUR TASK ---
Study the hard errors and each agent's exclusive solve history.
What type of problem is no one solving alone? What expert is missing?

Define that expert yourself.

Output in JSON. Keep system_prompt under 3 sentences:
{
    "persona_name": "...",
    "system_prompt": "...",
    "strengths": "..."
}
"""
)

