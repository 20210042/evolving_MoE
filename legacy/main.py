import argparse
import json
import os
from tqdm import tqdm
from src.utils.llm import LLMService
from src.agents.base import Agent
from src.pipelines.baselines import RawPipeline, SelfRefinePipeline
from src.pipelines.ours import PersonaRefinePipeline, OracleInitPipeline, InitDebatePipeline, CriticDebatePipeline
from src.pipelines.ours_v2 import GMRoutingPipeline
from src.data.loader import get_dataset

def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Persona Experiment Runner")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-32B-Instruct", help="Model path via HuggingFace/vLLM")

    parser.add_argument("--dataset", type=str, required=True, choices=["humaneval", "mbpp", "math", "ds1000", "livecodebench"], help="Dataset name")

    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument("--pipeline", type=str, required=True, choices=["raw", "self-refine", "ours", "oracle-init", "init-debate", "critic-debate", "ours_v2", "static-upper-bound"], help="Pipeline type")
    parser.add_argument("--persona", type=str, default="random", help="Persona strategy for 'ours' pipeline (random/specific_name)")
    parser.add_argument("--roster_path", type=str, default="results/gm_roster.json", help="Path to the GM scout roster for ours_v2")
    parser.add_argument(
        "--jina_router_checkpoint",
        type=str,
        default=None,
        help="If set (ours_v2), route with fine-tuned Jina from this directory instead of LLM Manager",
    )
    parser.add_argument("--max_iterations", type=int, default=4, help="Max Refine iterations")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSONL file path")
    parser.add_argument("--limit", type=int, default=-1, help="Limit number of samples for debugging")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature")
    parser.add_argument("--data_dir", type=str, default="/home/jaehoonjeong/data/MultiAgent/Data", help="Local data directory")
    
    args = parser.parse_args()

    # 1. Load Data
    print(f"Loading dataset: {args.dataset}")
    data = get_dataset(args.dataset, split=args.split, local_dir=args.data_dir)
    if args.limit > 0:
        data = data[:args.limit]
        print(f"Limiting to {args.limit} samples.")

    # 2. Init Model
    print(f"Initializing Model: {args.model}")
    llm = LLMService(model_name=args.model, mode="vllm") # Defaulting to vLLM
    agent = Agent(llm_service=llm)

    # 3. Init Pipeline
    print(f"Initializing Pipeline: {args.pipeline}")
    
    # Handle persona defaults per pipeline to avoid the "random" pitfall for Oracle
    persona = args.persona
    if args.pipeline == "oracle-init" and persona == "random":
        persona = "oracle"
    
    if args.pipeline == "raw":
        pipeline = RawPipeline(agent, domain=data[0]["domain"])
    elif args.pipeline == "self-refine":
        pipeline = SelfRefinePipeline(agent, domain=data[0]["domain"], max_iterations=args.max_iterations)
    elif args.pipeline == "ours":
        pipeline = PersonaRefinePipeline(agent, domain=data[0]["domain"], persona=persona, max_iterations=args.max_iterations)
    elif args.pipeline == "oracle-init":
        pipeline = OracleInitPipeline(agent, domain=data[0]["domain"], persona=persona)
    elif args.pipeline == "init-debate":
        pipeline = InitDebatePipeline(agent, domain=data[0]["domain"], persona=persona)
    elif args.pipeline == "critic-debate":
        pipeline = CriticDebatePipeline(agent, domain=data[0]["domain"], persona=persona)
    elif args.pipeline == "ours_v2":
        pipeline = GMRoutingPipeline(
            agent,
            scouting_report_path=args.roster_path,
            domain=data[0]["domain"],
            jina_router_checkpoint=args.jina_router_checkpoint,
        )
    elif args.pipeline == "static-upper-bound":
        from src.pipelines.ours import StaticUpperBoundPipeline
        pipeline = StaticUpperBoundPipeline(agent, domain=data[0]["domain"], roster_path=args.roster_path)
    
    # 4. Run Loop
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # RESUME LOGIC
    existing_ids = set()
    if os.path.exists(args.output_file):
        print(f"Output file found: {args.output_file}. Checking for existing items...")
        with open(args.output_file, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line)
                        existing_ids.add(item["id"])
                    except json.JSONDecodeError:
                        pass
        print(f"Found {len(existing_ids)} existing items. Resuming...")
    
    # Filter data
    original_len = len(data)
    data = [item for item in data if item["id"] not in existing_ids]
    print(f"Skipping {len(existing_ids)} items. Remaining: {len(data)}")
    
    with open(args.output_file, "a") as f:  # Append mode
        for item in tqdm(data):
            try:
                result = pipeline.run(item)
                # Save metadata
                result["dataset"] = args.dataset
                result["pipeline"] = args.pipeline
                if args.pipeline in ["ours", "oracle-init"]:
                    result["persona"] = result.get("selected_persona", args.persona)
                else:
                    result["persona"] = None
                
                # Write immediately (streaming)
                f.write(json.dumps(result) + "\n")
                f.flush()
            except Exception as e:
                print(f"Error processing item {item['id']}: {e}")
                import traceback
                traceback.print_exc()

    print(f"Experiment finished. Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
