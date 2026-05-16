import argparse
import json
import os
import logging
from tqdm import tqdm

from src.data.loader import get_dataset
from src.utils.llm import LLMService
from src.agents.base import Agent
from src.evaluation.scorer import evaluate_code_score, extract_helper_code
from src.prompts import baseline_prompts
from src.utils.helpers import extract_code_block, check_stop_condition

"""
Phase 7 Data Generation:
Generates the Multi-Label Target (Pass/Fail vector for K static personas)
for each problem in the training set to train an explicit Encoder Router.
"""

STATIC_ROSTER = [
    {"id": "c_1", "name": "General_Senior_Dev", "system_prompt": "You are a senior developer. Focus on clean code."},
    {"id": "c_2", "name": "Algorithmic_Expert", "system_prompt": "You are an algorithms expert. Focus on code structure constraints."},
    {"id": "c_3", "name": "QA_Red_Team", "system_prompt": "You are a QA engineer. Look for missed edge cases and constraints."},
    {"id": "c_4", "name": "CP_Master", "system_prompt": "You are an elite Competitive Programmer. Optimize for DP, Greedy, and avoid TLE/MLE."}
]

def score_item(item: dict, prediction_code: str) -> float:
    domain = item.get("domain", "coding")
    ground_truth = item.get("ground_truth")
    test_code = item.get("test_code") or item.get("test") or ""
    if isinstance(item.get("test_list"), list):
        test_code = "\n".join(item["test_list"])
    entry_point = item.get("entry_point")
    
    helper_code = extract_helper_code(ground_truth) if domain == "coding" else ""
    if domain == "coding" and "instruction" in item:
        instr_helper = extract_helper_code(item["instruction"])
        if instr_helper:
            helper_code = instr_helper + "\n" + helper_code
            
    return evaluate_code_score(prediction_code, test_code, entry_point, None, helper_code)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="livecodebench")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    parser.add_argument("--output_file", type=str, default="results/encoder_training_data.jsonl")
    parser.add_argument("--data_dir", type=str, default="/home/jaehoonjeong/data/MultiAgent/Data")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    
    data = get_dataset(args.dataset, split=args.split, local_dir=args.data_dir)
    if args.limit > 0:
        data = data[:args.limit]
        
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # We are generating fresh labels using the 4-iteration setup.
    # We will overwrite or ignore old existing trash labels.
    # To append safely without mixing trash labels, we should ideally write to a new file 
    # or just trust the user wants them appended/overwritten. Let's just run all data.
    logging.info(f"Generating NEW 4-iteration labels for {len(data)} problems.")
    
    llm = LLMService(model_name=args.model, mode="vllm")
    agent = Agent(llm_service=llm)
    
    # Overwrite the file to ensure we don't mix old 1-iteration trash labels with new 4-iteration labels
    with open(args.output_file, 'w') as f:
        for item in tqdm(data, desc="Evaluating Roster"):
            problem_id = item["id"]
            instruction = item.get("instruction", "")
            
            # Baseline Data
            sys_prompt = baseline_prompts.CODING_GEN_SYSTEM if item.get("domain", "coding") == "coding" else baseline_prompts.MATH_GEN_SYSTEM
            user_prompt_template = baseline_prompts.CODING_GEN_USER if item.get("domain", "coding") == "coding" else baseline_prompts.MATH_GEN_USER
            
            baseline_msg = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt_template.format(instruction=instruction)}
            ]
            baseline_raw = agent.chat(baseline_msg, temperature=0.0)
            baseline_code = extract_code_block(baseline_raw) or baseline_raw
            
            refine_user_prompt = (
                f"Instruction: {instruction}\n\nHere is a baseline code draft:\n```python\n{baseline_code}\n```\n\n"
                "Fix bugs. Output ONLY valid Python code inside a single ```python ... ``` block."
            )
            
            # Target Labels [0,1,0,1]
            labels = [0] * len(STATIC_ROSTER)
            
            for idx, persona in enumerate(STATIC_ROSTER):
                sys_prompt_persona = persona["system_prompt"]
                current_code = baseline_code
                
                for i in range(4):
                    # Persona as Critic
                    critic_user_prompt = (
                        f"Review the following code.\n\n"
                        f"Problem:\n{instruction}\n\n"
                        f"Code:\n{current_code}\n\n"
                        "1. Identify any syntax errors or bugs.\n"
                        "2. Check if it solves the problem correctly.\n"
                        "3. Assess the efficiency.\n\n"
                        "Output your review in the following format:\n"
                        "Feedback: [Your detailed feedback]"
                    )
                    
                    critic_msg = [
                        {"role": "system", "content": sys_prompt_persona},
                        {"role": "user", "content": critic_user_prompt}
                    ]
                    feedback = agent.chat(critic_msg, temperature=0.7)
                    if check_stop_condition(feedback):
                        break
                        
                    # General Expert as Refiner
                    refine_user_prompt = (
                        f"Refine the code based on the feedback.\n\n"
                        f"Problem:\n{instruction}\n\n"
                        f"Previous Code:\n{current_code}\n\n"
                        f"Feedback:\n{feedback}\n\n"
                        "Return the improved code in the following format:\n"
                        "```python\n[CODE]\n```"
                    )
                    
                    ref_msg = [
                        {"role": "system", "content": baseline_prompts.CODING_GEN_SYSTEM},
                        {"role": "user", "content": refine_user_prompt}
                    ]
                    ref_raw = agent.chat(ref_msg, temperature=0.0)
                    current_code = extract_code_block(ref_raw) or ref_raw
                
                score = score_item(item, current_code)
                if score >= 100.0:
                    labels[idx] = 1
                    
            output_record = {
                "problem_id": problem_id,
                "instruction": instruction,
                "labels": labels,
                "label_map": {persona["id"]: labels[idx] for idx, persona in enumerate(STATIC_ROSTER)},
                "num_solvers": sum(labels)
            }
            f.write(json.dumps(output_record) + "\n")
            f.flush()

if __name__ == "__main__":
    main()
