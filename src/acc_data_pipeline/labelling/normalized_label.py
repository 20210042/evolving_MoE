"""Batch label-normalization and missing-label filling script.

Maps raw original_domain tags into a fixed canonical taxonomy using aliases/regex rules, then uses
LiteLLM only for rows where labels are missing or unreliable."""

import re
import json
import csv
import pandas as pd
import ast
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tqdm import tqdm
from litellm import completion

# ==========================================
# 1. Configuration & Taxonomy Definition
# ==========================================

# LiteLLM 설정 (예: gpt-4o-mini 혹은 가성비 좋은 모델 추천)
MODEL_NAME = "gpt-4o-mini" 
LLM_MAX_WORKERS = 6
LLM_BATCH_SIZE = 128
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"

CANONICAL_LABELS = [
    "Dynamic Programming", "Graph Algorithms", "Tree Algorithms", "Greedy Algorithms",
    "Data Structures", "String Algorithms", "Math", "Number Theory", "Combinatorics",
    "Probability", "Geometry", "Binary Search", "Sorting", "Brute Force",
    "Backtracking", "Bit Manipulation", "Simulation", "Implementation",
    "Constructive Algorithms", "Recursion", "Game Theory", "Shortest Paths",
    "Union Find", "Range Queries", "Matrix", "Two Pointers", "Divide and Conquer", "Miscellaneous"
]

CRITIC_TAXONOMY_MAP = {
    "Math": "Quantitative Reasoning", "Number Theory": "Quantitative Reasoning",
    "Combinatorics": "Quantitative Reasoning", "Probability": "Quantitative Reasoning",
    "Geometry": "Quantitative Reasoning", "Game Theory": "Quantitative Reasoning",
    "Implementation": "Constructive Implementation", "Constructive Algorithms": "Constructive Implementation",
    "Simulation": "Constructive Implementation", "Recursion": "Constructive Implementation",
    "Data Structures": "Structured Data", "String Algorithms": "Structured Data",
    "Range Queries": "Structured Data", "Matrix": "Structured Data", "Union Find": "Structured Data",
    "Greedy Algorithms": "Greedy Strategy", "Sorting": "Greedy Strategy",
    "Bit Manipulation": "Greedy Strategy", "Binary Search": "Greedy Strategy",
    "Dynamic Programming": "State-Space Reasoning", "Graph Algorithms": "State-Space Reasoning",
    "Tree Algorithms": "State-Space Reasoning", "Shortest Paths": "State-Space Reasoning",
    "Brute Force": "State-Space Reasoning", "Backtracking": "State-Space Reasoning",
    "Two Pointers": "Greedy Strategy", # 혹은 알고리즘 특성에 따라 조정
    "Divide and Conquer": "State-Space Reasoning"
}

# ==========================================
# 2. Normalization Functions
# ==========================================

def regex_fallback(label_norm: str):
    label_norm = str(label_norm).lower()
    if re.search(r"segment tree|fenwick|range query|binary indexed tree|prefix sum", label_norm): return "Range Queries"
    if re.search(r"graph|dfs|bfs|topological|flow|matching", label_norm):
        if re.search(r"shortest|dijkstra|floyd|bellman", label_norm): return "Shortest Paths"
        return "Graph Algorithms"
    if re.search(r"two pointer|sliding window", label_norm): return "Two Pointers"
    if re.search(r"divide and conquer", label_norm): return "Divide and Conquer"
    if re.search(r"number theory|prime|modular|gcd|sieve|factor|lcm", label_norm): return "Number Theory"
    if re.search(r"tree|lca|euler tour", label_norm): return "Tree Algorithms"
    if re.search(r"dynamic|memo", label_norm): return "Dynamic Programming"
    if re.search(r"greedy|observation", label_norm): return "Greedy Algorithms"
    if re.search(r"data structure|array|stack|queue|heap|hash|list|set|map|stl", label_norm): return "Data Structures"
    if re.search(r"string|trie|kmp|suffix|regex|parser", label_norm): return "String Algorithms"
    if re.search(r"math|algebra|geometry|arithmetic|logic|polynomial|fft", label_norm):
        if re.search(r"geometry|convex", label_norm): return "Geometry"
        return "Math"
    if re.search(r"combination|counting", label_norm): return "Combinatorics"
    if re.search(r"probability|expected", label_norm): return "Probability"
    if re.search(r"binary search|search", label_norm): return "Binary Search"
    if re.search(r"sort", label_norm): return "Sorting"
    if re.search(r"brute|exhaustive|complete search", label_norm): return "Brute Force"
    if re.search(r"backtracking", label_norm): return "Backtracking"
    if re.search(r"bit|xor", label_norm): return "Bit Manipulation"
    if re.search(r"simulation|ad hoc|puzzle", label_norm): return "Simulation"
    if re.search(r"implementation", label_norm): return "Implementation"
    if re.search(r"constructive", label_norm): return "Constructive Algorithms"
    if re.search(r"recursion|recursive", label_norm): return "Recursion"
    if re.search(r"game", label_norm): return "Game Theory"
    if re.search(r"union find|disjoint|dsu", label_norm): return "Union Find"
    if re.search(r"matrix", label_norm): return "Matrix"
    return "Miscellaneous"

# ==========================================
# 3. LiteLLM Labeling Logic
# ==========================================

def get_labels_from_llm(problem_text):
    """LiteLLM을 사용하여 문제 텍스트로부터 멀티 라벨 추출"""
    prompt = f"""
Identify the relevant algorithm categories for the following programming problem.
Choose one or more labels from the provided canonical set ONLY.

[Canonical Labels]
{", ".join(CANONICAL_LABELS)}

[Problem Description]
{str(problem_text)[:2000]}

Return the result as a JSON object with a single key "labels" containing a list of strings.
Example: {{"labels": ["Dynamic Programming", "Math"]}}
"""
    try:
        response = completion(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format={ "type": "json_object" }
        )
        content = response.choices[0].message.content
        res_json = json.loads(content)
        
        # 안전한 JSON 파싱 로직
        labels = res_json.get("labels", [])
        if not labels and isinstance(res_json, dict) and len(res_json) > 0:
            # "labels" 키가 없으면 첫 번째 리스트 값을 가져옴
            labels = list(res_json.values())[0]
            
        if not isinstance(labels, list):
            labels = [labels]
        
        # Canonical Label에 있는 것만 필터링
        filtered_labels = [
            str(label).strip()
            for label in labels
            if str(label).strip() in CANONICAL_LABELS
        ]
        return filtered_labels or ["Miscellaneous"]

    except Exception as e:
        # 에러가 왜 나는지 정확히 확인하기 위해 터미널에 출력
        print(f"\n[API Error]: {e}")
        return ["Miscellaneous"]

# ==========================================
# 4. Main Processing Pipeline
# ==========================================

EMPTY_ORIGINAL_DOMAIN_VALUES = {"[]", '[""]', "['']", '""', "nan", "None", "null", ""}


def is_empty_original_domain(val):
    return str(val).strip() in EMPTY_ORIGINAL_DOMAIN_VALUES


def parse_list_cell(value):
    raw = str(value or "").strip()
    if not raw:
        return []
    try:
        parsed = ast.literal_eval(raw) if raw.startswith("[") else raw
    except (SyntaxError, ValueError):
        parsed = raw
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    return [str(parsed).strip()] if str(parsed).strip() else []


def should_retry_existing_row(row):
    if not is_empty_original_domain(row.get("original_domain")):
        return False
    normalized_labels = parse_list_cell(row.get("normalized_labels"))
    return (
        normalized_labels in ([], ["Miscellaneous"])
        or row.get("main_critic_category") == "Miscellaneous"
    )


def prepare_resume_output(output_csv_path, output_columns):
    output_csv_path = Path(output_csv_path)
    if not output_csv_path.exists() or output_csv_path.stat().st_size == 0:
        return set(), 0, 0

    csv.field_size_limit(sys.maxsize)
    temp_path = output_csv_path.with_suffix(output_csv_path.suffix + ".resume_tmp")
    completed = set()
    kept_count = 0
    retry_count = 0

    with output_csv_path.open("r", newline="", encoding="utf-8") as in_handle, temp_path.open(
        "w", newline="", encoding="utf-8"
    ) as out_handle:
        reader = csv.DictReader(in_handle)
        writer = csv.DictWriter(out_handle, fieldnames=output_columns)
        writer.writeheader()

        for row in reader:
            problem_id = row.get("problem_id")
            if not problem_id or not row.get("normalized_labels") or not row.get("main_critic_category"):
                retry_count += 1
                continue
            if should_retry_existing_row(row):
                retry_count += 1
                continue
            if problem_id in completed:
                continue

            writer.writerow({col: row.get(col, "") for col in output_columns})
            completed.add(problem_id)
            kept_count += 1

    temp_path.replace(output_csv_path)
    return completed, kept_count, retry_count

def process_pipeline(input_csv_path, output_csv_path):
    df = pd.read_csv(input_csv_path)
    
    # 1. 비어있는 라벨 판단 로직
    output_columns = list(df.columns)
    for col in ["normalized_labels", "critic_categories", "main_critic_category"]:
        if col not in output_columns:
            output_columns.append(col)

    main_category_counter = Counter()
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resume_existing = output_path.exists() and output_path.stat().st_size > 0
    completed_problem_ids, kept_existing_records, retry_existing_records = prepare_resume_output(
        output_path, output_columns
    )

    print("Starting Pipeline...")
    if resume_existing:
        print(f"Resuming from existing output: {output_path}")
        print(f"Already completed records: {len(completed_problem_ids)}")
        print(f"Kept existing records: {kept_existing_records}")
        print(f"Retrying existing records: {retry_existing_records}")

    def normalize_with_regex(raw_domain):
        normalized_list = []
        try:
            actual_labels = ast.literal_eval(raw_domain) if "[" in str(raw_domain) else [raw_domain]
            for lbl in actual_labels:
                normalized_list.append(regex_fallback(lbl))
        except:
            normalized_list = [regex_fallback(raw_domain)]
        return normalized_list

    write_mode = "a" if resume_existing else "w"
    written_records = 0
    skipped_records = 0

    with open(output_path, write_mode, newline="", encoding="utf-8") as out_f:
        writer = csv.writer(out_f)
        if not resume_existing:
            writer.writerow(output_columns)

        with ThreadPoolExecutor(max_workers=LLM_MAX_WORKERS) as executor:
            total_rows = len(df)
            for start in tqdm(range(0, total_rows, LLM_BATCH_SIZE), total=(total_rows + LLM_BATCH_SIZE - 1) // LLM_BATCH_SIZE):
                batch = df.iloc[start : start + LLM_BATCH_SIZE]
                futures = {}

                for idx, row in batch.iterrows():
                    problem_id = row.get("problem_id")
                    if problem_id in completed_problem_ids:
                        continue
                    raw_domain = row["original_domain"]
                    if is_empty_original_domain(raw_domain):
                        futures[idx] = executor.submit(get_labels_from_llm, row["problem"])

                for idx, row in batch.iterrows():
                    problem_id = row.get("problem_id")
                    if problem_id in completed_problem_ids:
                        skipped_records += 1
                        continue
                    raw_domain = row["original_domain"]
                    if idx in futures:
                        normalized_list = futures[idx].result()
                    else:
                        normalized_list = normalize_with_regex(raw_domain)

                    final_normalized = list(set(normalized_list))
                    critic_cats = list(set([CRITIC_TAXONOMY_MAP.get(l, "Miscellaneous") for l in final_normalized]))
                    main_critic = critic_cats[0] if critic_cats else "Miscellaneous"
                    main_category_counter[main_critic] += 1

                    row_dict = row.to_dict()
                    row_dict["normalized_labels"] = str(final_normalized)
                    row_dict["critic_categories"] = str(critic_cats)
                    row_dict["main_critic_category"] = main_critic
                    writer.writerow([row_dict.get(col, "") for col in output_columns])
                    completed_problem_ids.add(problem_id)
                    written_records += 1
                    out_f.flush()

    # 최종 분포 출력
    print("\n" + "="*30)
    print("Final Critic Distribution (Main)")
    print("="*30)
    print(pd.Series(main_category_counter).sort_values(ascending=False))
    print(f"Skipped existing records: {skipped_records}")
    print(f"New records written: {written_records}")

    return main_category_counter

# 실행부
# if __name__ == "__main__":
#    final_df = process_pipeline("your_data.csv", "labeled_output.csv")
