from string import Template

MANAGER_PROMPT = Template(
    """You are the General Manager (Router) of a software engineering team.
Your task is to analyze the following programming problem and select exactly one expert from your current roster who is best suited to solve it.

=== Current Roster Scouting Report ===
$scouting_report

=== Problem Description ===
$problem_description

Routing guidelines:
- Pick the critic whose strengths most specifically match the problem (e.g. subarray/indexing → array; parsing/text → string; DP/table → dp; graph/tree → graph; formulas/modulo/coordinates → math_geom).
- If several critics could apply, choose the most specific match, not the most general.
- If still ambiguous, prefer the critic whose strengths mention the problem's dominant algorithm type.

You MUST output your selection in valid JSON format exactly as follows:
{
    "selected_critic_id": "the_id_of_the_chosen_critic"
}
Do not output any other text or explanation. Only the JSON object.
"""
)

META_AGENT_PROMPT = Template(
    """You are the Head Scouter for an AI Competitive Programming Team.

Your mission is to analyze recent failures and define a specialist persona to fill the technical gaps identified in those failures.

=== FAILURE DIAGNOSIS (Hard Errors) ===
$hard_errors

=== CURRENT ROSTER (Active Personnel) ===
$current_roster

--- SCOUTING ORDERS ---
1. DIAGNOSE: Analyze the "Hard Errors" deeply. Determine if failures are due to algorithmic complexity (TLE), edge case handling, or input/output formatting.
2. GAP ANALYSIS: Identify why the current roster failed. Determine what specific technical expertise or focus is missing.
3. NON-REDUNDANCY (CRITICAL): Do NOT propose a persona whose expertise overlaps substantially with any current roster member. Read each member's strengths before writing.
4. COMPLEMENTARITY: Propose expertise for failure modes that existing members do not cover. If a similar domain already exists, choose a different angle or return a narrowly scoped variant only if hard errors clearly require it.
5. NAMING: Avoid reusing init domain keywords already present in roster ids or names (array, string, dp, graph, math, geom) unless the new persona is genuinely orthogonal.

Based on this, define a new specialist persona in the following JSON format:
{
    "persona_name": "A clear, descriptive name (e.g., DP_Optimization_Specialist)",
    "system_prompt": "A detailed system prompt defining the persona's technical expertise and focus for solving the identified failures.",
    "strengths": "Short description of specific expertise and problem types to solve",
    "custom_refine_prompt_template": "A precise instruction (1-2 sentences) dictating exactly what to check for in the candidate code.",
    "gap_not_covered_by": ["id1", "id2"]
}
"""
)
