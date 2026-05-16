from src.agents.base import Agent
from src.pipelines.baselines import BasePipeline
from src.prompts import baseline_prompts, personas
import random
import os

class PersonaRefinePipeline(BasePipeline):
    def __init__(self, agent: Agent, domain: str = "coding", persona: str = "random", max_iterations: int = 2):
        super().__init__(agent, domain)
        self.persona = persona
        self.max_iterations = max_iterations

    def run(self, input_item: dict):
        instruction = input_item.get("instruction") or input_item.get("prompt") or input_item.get("problem")
        starter_code = input_item.get("starter_code")
        if starter_code:
            instruction = f"{instruction}\n\nStarter Code:\n```python\n{starter_code}\n```"
        
        # MATH dataset typically has a 'type' field (e.g., 'Algebra'). If not present, default to 'Mathematics'
        topic = input_item.get("type", "Mathematics") 
        
        is_qwen3 = "qwen3" in self.agent.llm.model_name.lower()
        
        if self.domain == "coding":
            if is_qwen3:
                # Optimized Qwen3 LCB/Coding Gen
                from src.prompts import qwen3_lcb
                gen_msgs = [
                    {"role": "system", "content": qwen3_lcb.QWEN3_LCB_SYSTEM},
                    {"role": "user", "content": qwen3_lcb.QWEN3_LCB_USER_TEMPLATE.format(instruction=instruction)}
                ]
            else:
                gen_msgs = [{"role": "system", "content": baseline_prompts.CODING_GEN_SYSTEM}, 
                            {"role": "user", "content": baseline_prompts.CODING_GEN_USER.format(instruction=instruction)}]
        else:
            gen_msgs = [{"role": "system", "content": baseline_prompts.MATH_GEN_SYSTEM}, 
                        {"role": "user", "content": baseline_prompts.MATH_GEN_USER.format(instruction=instruction)}]
            
        # Generator: Temperature 0.0
        current_output = self.agent.chat(gen_msgs, temperature=0.0)
        history = [{"step": "initial", "output": current_output}]

        # 2. Select Persona (ONCE per task)
        if self.persona == "random":
            if self.domain == "coding" or self.domain == "ds":
                available = ["senior_dev", "qa_red_team", "code_grader", "cp_master"]
            else:
                available = ["professor", "math_tutor", "reasoning_assistant"]
            selected_persona = random.choice(available)
        else:
            selected_persona = self.persona

        for i in range(self.max_iterations):

            # 3. Persona-based Critic (Zero-Shot, User-Defined)
            critic_prompts = personas.get_persona_prompts(self.domain, selected_persona, instruction, current_output, topic, is_qwen3=is_qwen3)
            
            critic_msgs = [
                {"role": "system", "content": critic_prompts["system"]}, 
                {"role": "user", "content": critic_prompts["user"]}
            ]
            
            feedback = self.agent.chat(critic_msgs)
            
            # Stop Condition (Uses improved BasePipeline logic)
            stop = self.check_stop_condition(feedback)
            history.append({"step": f"critic_{i}", "persona": selected_persona, "feedback": feedback, "stop_signal": stop})

            if stop:
                print(f"Stopping early for Persona {selected_persona} due to stop signal.")
                break

            # 4. Refine
            if self.domain == "coding" or self.domain == "ds":
                ref_system = baseline_prompts.CODING_GEN_SYSTEM + " Always provide the COMPLETE code including all necessary imports."
                ref_user = f"""Refine the code based on the feedback. Ensure you include all necessary imports (like typing.List) in the output.
[Task]
{instruction}

[Previous Code]
{current_output}

[Feedback]
{feedback}

Return the COMPLETE improved code in the following format:
```python
[CODE]
```"""
            else:
                ref_system = "You are a helpful math assistant."
                ref_user = f"""Refine the solution based on the feedback.
[Task]
{instruction}

[Previous Solution]
{current_output}

[Feedback]
{feedback}

Provide the corrected solution.
Answer format:
Final Answer: [ANSWER]"""

            ref_msgs = [{"role": "system", "content": ref_system}, {"role": "user", "content": ref_user}]
            
            # Generator: Temperature 0.0
            refined_output = self.agent.chat(ref_msgs, temperature=0.0)
            
            current_output = refined_output
            history.append({"step": f"refine_{i}", "output": current_output})

        return {
            "id": input_item.get("id"),
            "initial_output": history[0]["output"],
            "final_output": current_output,
            "history": history,
            "selected_persona": selected_persona
        }

class OracleInitPipeline(BasePipeline):
    """
    1-Shot Generation Pipeline that injects the Oracle persona into the initial response prompt.
    Does not use critics or refinement.
    """
    def __init__(self, agent: Agent, domain: str = "coding", persona: str = "oracle"):
        super().__init__(agent, domain)
        self.persona = persona

    def infer_persona(self, instruction: str) -> str:
        instruction_lower = instruction.lower()
        
        # Original Categorization Logic from our Heatmap Analysis
        
        def get_category(text):
            if any(kw in text for kw in ['graph', 'tree', 'node', 'edge']):
                return 'Graph'
            if any(kw in text for kw in ['math', 'geometry', 'calculate', 'equation', 'probability']):
                return 'Math'
            if any(kw in text for kw in ['string', 'substring', 'character', 'regex', 'palindrome']):
                return 'String'
            if any(kw in text for kw in ['matrix', 'grid', 'array', 'list', 'dictionary', 'map', 'hash', 'element', 'index']):
                return 'Data Structure'
            if any(kw in text for kw in ['minimum', 'maximum', 'maximize', 'minimize', 'greedy', 'longest', 'shortest']):
                return 'Greedy'
            if any(kw in text for kw in ['sort', 'search', 'binary search', 'find']):
                return 'Sort/Search'
            return 'Other'

        category = get_category(instruction_lower)
        
        # Adaptive Routing based on Heatmap Champions
        # Champion for Graph, Math -> code_grader
        # Champion for String -> code_grader (Biased)
        if category in ['Math', 'Graph', 'String']:
            return 'code_grader'
            
        # Champion for Data Structure, Greedy, Sort/Search -> senior_dev
        # Other / Unknown -> senior_dev (defaulting to general programming)
        return 'senior_dev'

    def run(self, input_item: dict):
        instruction = input_item.get("instruction") or input_item.get("prompt") or input_item.get("problem")
        starter_code = input_item.get("starter_code")
        if starter_code:
            instruction = f"{instruction}\n\nStarter Code:\n```python\n{starter_code}\n```"
        
        topic = input_item.get("type", "Mathematics")
        
        # 1. Select Persona
        if self.persona == "oracle":
            selected_persona = self.infer_persona(instruction)
        else:
            selected_persona = self.persona

        # 2. Initial Generation (Backbone Generator)
        if self.domain == "coding":
            gen_system = baseline_prompts.CODING_GEN_SYSTEM
            gen_user = baseline_prompts.CODING_GEN_USER.format(instruction=instruction)
        else:
            gen_system = baseline_prompts.MATH_GEN_SYSTEM
            gen_user = baseline_prompts.MATH_GEN_USER.format(instruction=instruction)

        # 2. Initial Generation (Backbone Generator): Temperature 0.0
        current_output = self.agent.chat([{"role": "system", "content": gen_system}, {"role": "user", "content": gen_user}], temperature=0.0)
        history = [{"step": "initial", "output": current_output, "persona": selected_persona}]

        # 3. 4-Iteration Persona-as-Critic Loop
        for i in range(4):
            # A. Critique Pass - Selected Persona as Critic
            critic_prompts = personas.get_persona_prompts(self.domain, selected_persona, instruction, current_output, topic)
            critic_msgs = [
                {"role": "system", "content": critic_prompts["system"]}, 
                {"role": "user", "content": critic_prompts["user"]}
            ]
            feedback = self.agent.chat(critic_msgs)
            
            stop = self.check_stop_condition(feedback)
            history.append({"step": f"critic_{i}", "persona": selected_persona, "feedback": feedback, "stop_signal": stop})
            if stop: break

            # B. Revision Pass - Backbone Generator as Refiner: Temperature 0.0
            if self.domain == "coding":
                ref_system = baseline_prompts.CODING_GEN_SYSTEM + " Always provide the COMPLETE code including all necessary imports."
                ref_user = f"Refine the code based on the feedback.\n[Task]\n{instruction}\n\n[Previous Code]\n{current_output}\n\n[Feedback]\n{feedback}\n\nReturn the COMPLETE improved code in the following format:\n```python\n[CODE]\n```"
            else:
                ref_system = "You are a helpful math assistant." # Standardizing later if needed
                ref_user = f"Refine the solution based on the feedback.\n[Task]\n{instruction}\n\n[Previous Solution]\n{current_output}\n\n[Feedback]\n{feedback}\n\nProvide the corrected final solution.\nAnswer format:\nFinal Answer: [ANSWER]"

            ref_msgs = [{"role": "system", "content": ref_system}, {"role": "user", "content": ref_user}]
            current_output = self.agent.chat(ref_msgs, temperature=0.0)
            history.append({"step": f"refine_{i}", "output": current_output})

        return {
            "id": input_item.get("id"),
            "initial_output": history[0]["output"],
            "final_output": current_output,
            "history": history,
            "selected_persona": selected_persona
        }

class InitDebatePipeline(BasePipeline):
    """
    3 Personas generate initial code independently.
    A Judge persona selects or synthesizes the best final code.
    """
    def __init__(self, agent: Agent, domain: str = "coding", persona: str = "debate"):
        super().__init__(agent, domain)
        self.personas = ["senior_dev", "code_grader", "qa_red_team"]

    def run(self, input_item: dict):
        instruction = input_item.get("instruction") or input_item.get("prompt") or input_item.get("problem")
        starter_code = input_item.get("starter_code")
        if starter_code:
            instruction = f"{instruction}\n\nStarter Code:\n```python\n{starter_code}\n```"
        
        topic = input_item.get("type", "Mathematics")
        
        # 1. Parallel Generation
        generated_codes = {}
        for p in self.personas:
            sys_prompt = personas.get_generation_system_prompt(self.domain, p, topic)
            if self.domain == "coding":
                sys_prompt += " Always provide the COMPLETE code including all necessary imports."
                user_prompt = f"""Write a Python function to solve the following problem. Ensure you include all necessary imports (like typing.List) in the output.

### Task
Problem:
{instruction}

Return the COMPLETE code in the following format:
```python
[CODE]
```"""
            else:
                user_prompt = baseline_prompts.MATH_GEN_USER.format(instruction=instruction)
            
            gen_msgs = [
                {"role": "system", "content": sys_prompt}, 
                {"role": "user", "content": user_prompt}
            ]
            # Generator: Temperature 0.0
            generated_codes[p] = self.agent.chat(gen_msgs, temperature=0.0)
            
        # 2. Judge / Aggregation
        if self.domain == "coding":
            judge_sys = baseline_prompts.CODING_GEN_SYSTEM + " Your job is to review multiple proposed code solutions and synthesize the absolute best, most robust, and optimal final code. Always provide the COMPLETE code including all necessary imports."
            judge_user = f"""Review the following 3 proposed solutions for the given problem. Synthesize the best aspects of each, fix any bugs, and provide the absolute best final code. Ensure you include all necessary imports (like typing.List) in the output.

### Task
Problem:
{instruction}

### Proposed Solution 1 (Senior Dev)
{generated_codes['senior_dev']}

### Proposed Solution 2 (Code Grader)
{generated_codes['code_grader']}

### Proposed Solution 3 (QA Red Team)
{generated_codes['qa_red_team']}

Return the COMPLETE final synthesized code in the following format:
```python
[CODE]
```"""
        else:
            judge_sys = "You are an expert Lead Mathematician. Your job is to review multiple proposed solutions and synthesize the absolute best final mathematical answer."
            judge_user = f"""Review the following 3 proposed solutions for the given problem. Synthesize the correct logic and provide the absolute best final answer.

### Task
Problem:
{instruction}

### Proposed Solution 1
{generated_codes['senior_dev']}

### Proposed Solution 2
{generated_codes['code_grader']}

### Proposed Solution 3
{generated_codes['qa_red_team']}

Provide the corrected final solution.
Answer format:
Final Answer: [ANSWER]"""

        judge_msgs = [{"role": "system", "content": judge_sys}, {"role": "user", "content": judge_user}]
        final_output = self.agent.chat(judge_msgs, temperature=0.0)
        
        history = [
            {"step": "init_senior_dev", "output": generated_codes['senior_dev']},
            {"step": "init_code_grader", "output": generated_codes['code_grader']},
            {"step": "init_qa_red_team", "output": generated_codes['qa_red_team']},
            {"step": "judge_synthesis", "output": final_output}
        ]

        return {
            "id": input_item.get("id"),
            "initial_output": generated_codes['senior_dev'], # arbitrary primary
            "final_output": final_output,
            "history": history,
            "selected_persona": "debate_judge"
        }
class CriticDebatePipeline(BasePipeline):
    """
    1. Generates 1 initial baseline code.
    2. 3 Personas independently critique the baseline.
    3. 3 Personas debate: refine their own critique by reviewing the other two.
    4. Aggregator synthesizes a final refinement prompt to fix the code.
    """
    def __init__(self, agent: Agent, domain: str = "coding", persona: str = "debate"):
        super().__init__(agent, domain)
        self.personas = ["senior_dev", "code_grader", "qa_red_team"]

    def run(self, input_item: dict):
        instruction = input_item.get("instruction") or input_item.get("prompt") or input_item.get("problem")
        starter_code = input_item.get("starter_code")
        if starter_code:
            instruction = f"{instruction}\n\nStarter Code:\n```python\n{starter_code}\n```"
        
        topic = input_item.get("type", "Mathematics")
        
        # 1. Initial Generation (Baseline)
        if self.domain == "coding":
            gen_msgs = [{"role": "system", "content": baseline_prompts.CODING_GEN_SYSTEM}, 
                        {"role": "user", "content": baseline_prompts.CODING_GEN_USER.format(instruction=instruction)}]
        else:
            gen_msgs = [{"role": "system", "content": baseline_prompts.MATH_GEN_SYSTEM}, 
                        {"role": "user", "content": baseline_prompts.MATH_GEN_USER.format(instruction=instruction)}]
            
        # Generator: Temperature 0.0
        initial_output = self.agent.chat(gen_msgs, temperature=0.0)
        history = [{"step": "initial", "output": initial_output}]

        # 2. Parallel First Critique
        critiques_round1 = {}
        for p in self.personas:
            critic_prompts = personas.get_persona_prompts(self.domain, p, instruction, initial_output, topic)
            critic_msgs = [
                {"role": "system", "content": critic_prompts["system"]}, 
                {"role": "user", "content": critic_prompts["user"]}
            ]
            critiques_round1[p] = self.agent.chat(critic_msgs)
            history.append({"step": f"critique1_{p}", "feedback": critiques_round1[p]})

        # 3. Parallel Second Critique (Debate)
        critiques_round2 = {}
        for p in self.personas:
            other_personas = [op for op in self.personas if op != p]
            critic_prompts = personas.get_persona_prompts(self.domain, p, instruction, initial_output, topic)
            
            debate_sys = critic_prompts["system"] + " You must now refine your critique based on the opinions of your colleagues."
            
            debate_user = f"""You previously provided feedback on a proposed solution. Two of your peers have also evaluated the same solution. 
Review their perspectives and provide a FINAL, synthesized critique that addresses their points either by respectfully agreeing or disagreeing. Focus entirely on the code's flaws.

### Your Initial Feedback
{critiques_round1[p]}

### Peer 1 ({other_personas[0]}) Feedback
{critiques_round1[other_personas[0]]}

### Peer 2 ({other_personas[1]}) Feedback
{critiques_round1[other_personas[1]]}

Write your final consolidated critique:"""
            
            debate_msgs = [
                {"role": "system", "content": debate_sys}, 
                {"role": "user", "content": debate_user}
            ]
            critiques_round2[p] = self.agent.chat(debate_msgs)
            history.append({"step": f"critique2_debate_{p}", "feedback": critiques_round2[p]})

        # 4. Aggregation and Refinement
        aggregated_feedback = f"""### Senior Dev Final Review:
{critiques_round2['senior_dev']}

### Code Grader Final Review:
{critiques_round2['code_grader']}

### QA Red Team Final Review:
{critiques_round2['qa_red_team']}"""

        if self.domain == "coding" or self.domain == "ds":
            ref_system = baseline_prompts.CODING_GEN_SYSTEM + " Always provide the COMPLETE code including all necessary imports."
            ref_user = f"""A panel of experts debated the quality of your code and provided the following consolidated feedback. Refine the code addressing their major concerns. Ensure you include all necessary imports (like typing.List) in the output.
[Task]
{instruction}

[Previous Code]
{initial_output}

[Experts Debate Feedback]
{aggregated_feedback}

Return the COMPLETE improved code in the following format:
```python
[CODE]
```"""
        else:
            ref_system = "You are a helpful math assistant."
            ref_user = f"""Refine the solution based on the feedback.
[Task]
{instruction}

[Previous Solution]
{initial_output}

[Experts Debate Feedback]
{aggregated_feedback}

Provide the corrected solution.
Answer format:
Final Answer: [ANSWER]"""

        ref_msgs = [{"role": "system", "content": ref_system}, {"role": "user", "content": ref_user}]
        # Generator: Temperature 0.0
        final_output = self.agent.chat(ref_msgs, temperature=0.0)
        history.append({"step": "refine_final", "output": final_output})
        
        return {
            "id": input_item.get("id"),
            "initial_output": initial_output,
            "final_output": final_output,
            "history": history,
            "selected_persona": "debate_panel"
        }

class StaticUpperBoundPipeline(BasePipeline):
    """
    Measures the Static Upper Bound:
    Runs all 3 initial experts (Senior Dev, Code Grader, QA Red Team) on the same problem.
    The union of their success is the Upper Bound.
    """
    def __init__(self, agent: Agent, domain: str = "coding", roster_path: str = None):
        super().__init__(agent, domain)
        if roster_path and os.path.exists(roster_path):
            import json
            with open(roster_path, 'r') as f:
                self.roster = json.load(f)
        else:
            self.roster = [
                {"id": "senior_dev", "name": "Senior Software Engineer", "system_prompt": "You are a senior software engineer. Focus on clean, modular, and robust code for competitive programming."},
                {"id": "code_grader", "name": "Code Grader & Optimizer", "system_prompt": "You are a strict code grader. Focus on time/space complexity and identifying potential logical flaws that lead to incorrect answers."},
                {"id": "qa_red_team", "name": "QA Red Team", "system_prompt": "You are a member of the QA Red Team. Your goal is to break the code by finding unhandled edge cases, infinite loops, or memory limits."},
                {"id": "self_refine", "name": "Self-Refine", "system_prompt": "You are a helpful assistant."}
            ]

    def run(self, input_item: dict):
        instruction = input_item.get("instruction") or input_item.get("prompt") or input_item.get("problem")
        starter_code = input_item.get("starter_code")
        if starter_code:
            instruction = f"{instruction}\n\nStarter Code:\n```python\n{starter_code}\n```"

        is_qwen3 = "qwen3" in self.agent.llm.model_name.lower()
        from src.utils.helpers import extract_code_block, check_stop_condition

        # 1. Baseline Generation (Exactly match Ours v2 branching)
        if self.domain == "coding":
            if is_qwen3:
                gen_sys = baseline_prompts.LCB_GEN_SYSTEM
                gen_user = baseline_prompts.LCB_GEN_USER.format(instruction=instruction)
                baseline_msg = (
                    f"<|im_start|>system\n{gen_sys}\n<|im_end|>\n"
                    f"<|im_start|>user\n{gen_user}\n<|im_end|>\n"
                    f"<|im_start|>assistant\n```python\n"
                )
            else:
                baseline_msg = [
                    {"role": "system", "content": baseline_prompts.CODING_GEN_SYSTEM},
                    {"role": "user", "content": baseline_prompts.CODING_GEN_USER.format(instruction=instruction)}
                ]
        else:
            baseline_msg = [
                {"role": "system", "content": baseline_prompts.MATH_GEN_SYSTEM},
                {"role": "user", "content": baseline_prompts.MATH_GEN_USER.format(instruction=instruction)}
            ]

        baseline_res = self.agent.chat(baseline_msg, temperature=0.0)
        baseline_code = extract_code_block(baseline_res) or baseline_res
        
        results = []
        for player in self.roster:
            current_code = baseline_code
            player_history = []
            critic_sys_prompt = player.get('system_prompt', "You are a specialized code critic.")
            
            for i in range(4): # 4 iterations
                # A. Critique Pass (Exactly match Ours v2 prompt construction)
                if is_qwen3:
                    crit_msg = (
                        f"<|im_start|>system\n{critic_sys_prompt}\n<|im_end|>\n"
                        f"<|im_start|>user\n{baseline_prompts.CODING_CRITIC_USER.format(instruction=instruction, code=current_code)}\n<|im_end|>\n"
                        "<|im_start|>assistant\nFeedback: "
                    )
                else:
                    crit_msg = [
                        {"role": "system", "content": critic_sys_prompt},
                        {"role": "user", "content": (
                            f"Review the following code.\n\n"
                            f"Problem:\n{instruction}\n\n"
                            f"Code:\n{current_code}\n\n"
                            "1. Identify any syntax errors or bugs.\n"
                            "2. Check if it solves the problem correctly.\n"
                            "3. Assess the efficiency.\n\n"
                            "Output your review in the following format:\n"
                            "Feedback: [Your detailed feedback]"
                        )}
                    ]
                
                feedback = self.agent.chat(crit_msg)
                
                if check_stop_condition(feedback):
                    player_history.append({"iter": i, "stage": "critique", "feedback": feedback, "status": "stopped_early"})
                    break
                
                # B. Revision Pass (Exactly match Ours v2 prompt construction)
                if self.domain == "coding":
                    if is_qwen3:
                        refine_user_prompt = (
                            f"<|im_start|>system\n{baseline_prompts.CODING_REVISION_SYSTEM}\n<|im_end|>\n"
                            f"<|im_start|>user\n{baseline_prompts.CODING_REVISION_USER.format(instruction=instruction, code=current_code, feedback=feedback)}\n<|im_end|>\n"
                            f"<|im_start|>assistant\n```python\n"
                        )
                        final_code_raw = self.agent.chat(refine_user_prompt, temperature=0.0)
                    else:
                        gen_sys_prompt = baseline_prompts.CODING_GEN_SYSTEM
                        refine_user_prompt = (
                            f"Refine the code based on the feedback.\n\n"
                            f"Problem:\n{instruction}\n\n"
                            f"Previous Code:\n{current_code}\n\n"
                            f"Feedback:\n{feedback}\n\n"
                            "Return the improved code in the following format:\n"
                            "```python\n[CODE]\n```"
                        )
                        final_code_raw = self.agent.chat([
                            {"role": "system", "content": gen_sys_prompt},
                            {"role": "user", "content": refine_user_prompt}
                        ], temperature=0.0)
                else:
                    gen_sys_prompt = "You are a helpful math assistant."
                    refine_user_prompt = (
                        f"Refine the solution based on the feedback.\n\n"
                        f"Problem:\n{instruction}\n\n"
                        f"Previous Solution:\n{current_code}\n\n"
                        f"Feedback:\n{feedback}\n\n"
                        "Provide the corrected solution.\n"
                        "Answer format:\n"
                        "Final Answer: [ANSWER]"
                    )
                    final_code_raw = self.agent.chat([
                        {"role": "system", "content": gen_sys_prompt},
                        {"role": "user", "content": refine_user_prompt}
                    ], temperature=0.0)
                    
                current_code = extract_code_block(final_code_raw) or final_code_raw
                player_history.append({"iter": i, "stage": "revision", "code": current_code})
            
            results.append({
                "persona": player["id"],
                "final_output": current_code,
                "player_history": player_history
            })

        return {
            "id": input_item.get("id"),
            "initial_output": baseline_code,
            "final_output": results[0]["final_output"], 
            "all_persona_outputs": results, 
            "history": results
        }
