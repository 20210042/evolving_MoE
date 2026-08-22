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

MANAGER_QASC_PROMPT = Template(
    """You are the Router of a science multiple-choice team.
Assign the following question to exactly one specialist from your roster.

=== Current Roster ===
$scouting_report

=== Question ===
$problem_description

Pick the specialist most likely to answer correctly.
Output only valid JSON:
{
    "selected_expert_id": "the_id_of_the_chosen_expert"
}
"""
)

MANAGER_SNI_PROMPT = Template(
    """You are the Router of a general instruction-following team.
Assign the following task to exactly one specialist from your roster.

=== Current Roster ===
$scouting_report

=== Task ===
$problem_description

Pick the specialist most likely to produce the exact required output.
Output only valid JSON:
{
    "selected_expert_id": "the_id_of_the_chosen_expert"
}
"""
)

MANAGER_LEGAL_PROMPT = Template(
    """You are the Router of a Korean legal reasoning team.
Assign the following legal classification task to exactly one specialist from your roster.

=== Current Roster ===
$scouting_report

=== Task ===
$problem_description

Pick the specialist most likely to produce the exact required answer.
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

META_AGENT_QASC_PROMPT = Template(
    """You are the Head Scouter for an AI Science Multiple-Choice Team.

Every agent on the current roster failed the following questions:

=== HARD ERRORS ===
$hard_errors

=== CURRENT ROSTER ===
$current_roster

--- YOUR TASK ---
Look at these unsolved questions. What science specialist is missing from this roster?

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

META_AGENT_LEGAL_PROMPT = Template(
    """You are the Head Scouter for an AI Korean Legal Classification Team.

Every agent on the current roster failed the following legal tasks:

=== HARD ERRORS ===
$hard_errors

=== CURRENT ROSTER ===
$current_roster

--- YOUR TASK ---
Look at these unsolved legal tasks. What legal specialist is missing from this roster?

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

# SNI: 분해 축을 "주제"가 아니라 "태스크 유형/요구 능력"으로 못박는다 — 이게 arm B의
# 관측 대상이다. 나머지 구조(규칙 3개, JSON 스키마)는 QASC/LEGAL과 동일하게 유지한다.
META_AGENT_SNI_PROMPT = Template(
    """You are the Head Scouter for an AI General Instruction-Following Team.

Every agent on the current roster failed the following tasks:

=== HARD ERRORS ===
$hard_errors

=== CURRENT ROSTER ===
$current_roster

--- YOUR TASK ---
Look at these unsolved tasks. Note what *kind* of task each one is (what operation it
demands and what output format it requires), not merely what topic it is about.

What kind of task specialist is missing from this roster?

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

# Seed 20210017+: failure-mode scout. hard_errors include one failed attempt per
# problem; scout types the recurring *mistake* (not the topic) and mints an expert
# that avoids it. Lean: same rules as V1, strengths dropped.
META_AGENT_MATH_PROMPT_FAILURE = Template(
    """You are the Head Scouter for an AI Mathematics Team.

The whole roster failed these problems. Each is shown with one failed attempt:

=== HARD ERRORS (problem + a failed attempt) ===
$hard_errors

=== CURRENT ROSTER ===
$current_roster

--- YOUR TASK ---
What recurring mistake do these attempts share? Define one expert that avoids it.

Rules:
1. NON-REDUNDANCY (CRITICAL): Must be genuinely different from every current roster member.
2. ATOMICITY (CRITICAL): One expert, one focused identity — not a combination of multiple.
3. persona_name must NOT contain the word 'and'.

Output in JSON. Keep system_prompt under 3 sentences:
{
    "persona_name": "...",
    "system_prompt": "..."
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


# V3: 정체성(system) + 접근법(user) 분리. system_prompt = 정체성 1문장(누구인가),
# approach = 어떻게 푸는가(절차적 방법). strengths 없음. "expert" 등 priming 표현 미사용.
META_AGENT_MATH_PROMPT_V3 = Template(
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
For the problems no one is solving alone, design a new agent that would crack them.

Provide:
- "system_prompt": ONE sentence stating WHO this agent is — its identity / mindset.
- "approach": HOW this agent attacks such problems — the concrete procedure it follows:
  what it sets up first, which strategy it reaches for, how it verifies its result.

Output in JSON:
{
    "persona_name": "...",
    "system_prompt": "...",
    "approach": "..."
}
"""
)
