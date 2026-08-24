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

# SNI: **축을 지정하지 않는다.** 데이터가 실제로 (태스크, 도메인) 두 축인데 한쪽을 가리키면
# 답을 정해놓고 묻는 셈이다. 실패 사례만 주고 무엇을 공통점으로 보는지는 모델이 정하게 한다 —
# `fixes` 필드가 곧 "모델이 어떤 축을 봤는가"의 기록이고, 파일럿의 관측 대상이 그것이다.
# 정체성("You are a ... specialist")을 요구하지 않는다: 정체성은 문체만 바꾼다는 게
# acc·QASC·SNI에서 세 번 확인됐다(docs/REPORT_for_collab_sni_axis.md).
META_AGENT_SNI_PROMPT = Template(
    """You are improving an AI team. Each member works from its own system prompt.

Every member of the current team failed the cases below. Each case shows the task,
the input, the expected output, and what the team actually produced.

=== FAILED CASES ===
$hard_errors

=== SYSTEM PROMPTS ALREADY IN USE ===
$current_roster

--- YOUR TASK ---
Whatever these failures have in common, write one new system prompt that would have
produced the expected outputs.

Rules:
1. It must cover something the system prompts already in use do not.
2. One thing, stated concretely. Not a combination.
3. Keep it under 3 sentences.

Output in JSON:
{
    "prompt_name": "...",
    "system_prompt": "...",
    "fixes": "<what these failures had in common, one line>"
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
