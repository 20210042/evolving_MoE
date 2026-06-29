#!/usr/bin/env python3
"""Final deterministic labeling pass for execution-ready CSV rows.

Merges normalized labels with the execution-ready table, maps canonical labels into critic categories,
and writes final label/distribution artifacts without modifying the source normalized-label file."""

from __future__ import annotations

import argparse
import ast
import csv
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

INPUT_CSV = Path(
    "/home/minjikim/minji_link/code/benchmark/data/labelling/04_execution_ready_normalized_labels_163969.csv"
)
TAG_DISTRIBUTION_CSV = Path(
    "/home/minjikim/minji_link/code/benchmark/data/reports/tag_distribution.csv"
)
OUTPUT_DIR = Path("/home/minjikim/minji_link/code/benchmark/data/labelling")

CANONICAL_LABELS = [
    "Dynamic Programming",
    "Graph Algorithms",
    "Tree Algorithms",
    "Greedy Algorithms",
    "Data Structures",
    "String Algorithms",
    "Math",
    "Number Theory",
    "Combinatorics",
    "Probability",
    "Geometry",
    "Binary Search",
    "Sorting",
    "Brute Force",
    "Backtracking",
    "Bit Manipulation",
    "Simulation",
    "Implementation",
    "Constructive Algorithms",
    "Recursion",
    "Game Theory",
    "Shortest Paths",
    "Union Find",
    "Range Queries",
    "Matrix",
    "Two Pointers",
    "Divide and Conquer",
    "Miscellaneous",
]

CANONICAL_SET = set(CANONICAL_LABELS)
LOW_SIGNAL_FALLBACK_LABEL = "Implementation"

CRITIC_TAXONOMY_MAP = {
    "Math": "Quantitative Reasoning",
    "Number Theory": "Quantitative Reasoning",
    "Combinatorics": "Quantitative Reasoning",
    "Probability": "Quantitative Reasoning",
    "Geometry": "Quantitative Reasoning",
    "Game Theory": "Quantitative Reasoning",
    "Implementation": "Constructive Implementation",
    "Constructive Algorithms": "Constructive Implementation",
    "Simulation": "Constructive Implementation",
    "Recursion": "Constructive Implementation",
    "Data Structures": "Structured Data",
    "String Algorithms": "Structured Data",
    "Range Queries": "Structured Data",
    "Matrix": "Structured Data",
    "Union Find": "Structured Data",
    "Greedy Algorithms": "Greedy Strategy",
    "Sorting": "Greedy Strategy",
    "Bit Manipulation": "Greedy Strategy",
    "Binary Search": "Greedy Strategy",
    "Two Pointers": "Greedy Strategy",
    "Dynamic Programming": "State-Space Reasoning",
    "Graph Algorithms": "State-Space Reasoning",
    "Tree Algorithms": "State-Space Reasoning",
    "Shortest Paths": "State-Space Reasoning",
    "Brute Force": "State-Space Reasoning",
    "Backtracking": "State-Space Reasoning",
    "Divide and Conquer": "State-Space Reasoning",
    "Miscellaneous": "Constructive Implementation",
}

CRITIC_PRIORITY = [
    "State-Space Reasoning",
    "Structured Data",
    "Quantitative Reasoning",
    "Greedy Strategy",
    "Constructive Implementation",
]

EMPTY_ORIGINAL_DOMAIN_VALUES = {"[]", '[""]', "['']", '""', "nan", "None", "null", ""}

EXACT_ALIASES = {
    "ad hoc": "Simulation",
    "adhoc": "Simulation",
    "ap": "Math",
    "basic maths": "Math",
    "bfs": "Graph Algorithms",
    "bit": "Range Queries",
    "cg": "Geometry",
    "complete search": "Brute Force",
    "cpp": "Implementation",
    "dfs": "Graph Algorithms",
    "dfs and similar": "Graph Algorithms",
    "dp": "Dynamic Programming",
    "dsu": "Union Find",
    "fft": "Math",
    "gp": "Math",
    "java": "Implementation",
    "json": "Implementation",
    "maths": "Math",
    "misc": LOW_SIGNAL_FALLBACK_LABEL,
    "numpy": "Implementation",
    "sssp": "Shortest Paths",
    "stl": "Data Structures",
}

COMPACT_ALIASES = {
    "ahocorasick": "String Algorithms",
    "avltree": "Data Structures",
    "bellmanford": "Shortest Paths",
    "binaryindexedtree": "Range Queries",
    "binarylifting": "Tree Algorithms",
    "binarysearch": "Binary Search",
    "binarysearchtree": "Data Structures",
    "binarytree": "Tree Algorithms",
    "bitmanipulation": "Bit Manipulation",
    "bitwiseoperation": "Bit Manipulation",
    "bruteforce": "Brute Force",
    "chineseremaindertheorem": "Number Theory",
    "datastructure": "Data Structures",
    "datastructures": "Data Structures",
    "disjointset": "Union Find",
    "disjointsets": "Union Find",
    "disjointsetunion": "Union Find",
    "divideandconquer": "Divide and Conquer",
    "dynamicprogramming": "Dynamic Programming",
    "eulertheorem": "Number Theory",
    "eulertotientfunction": "Number Theory",
    "extendedeuclid": "Number Theory",
    "fastfouriertransform": "Math",
    "fenwicktree": "Range Queries",
    "fenwicktrees": "Range Queries",
    "floydwarshall": "Shortest Paths",
    "gametheory": "Game Theory",
    "graphalgos": "Graph Algorithms",
    "graphtheory": "Graph Algorithms",
    "greatestcommondivisor": "Number Theory",
    "grundynumbers": "Game Theory",
    "heavylightdecomposition": "Tree Algorithms",
    "kmpalgorithm": "String Algorithms",
    "linearsweep": "Geometry",
    "longestincsubsequence": "Dynamic Programming",
    "manachersalgorithm": "String Algorithms",
    "matrixexponentiation": "Matrix",
    "minimumspanningtree": "Graph Algorithms",
    "minimumspanningtrees": "Graph Algorithms",
    "modulararithmetic": "Number Theory",
    "modularexponentiation": "Number Theory",
    "numbertheoretictransfmtn": "Math",
    "numbertheory": "Number Theory",
    "priorityqueue": "Data Structures",
    "rabinkarpalgorithm": "String Algorithms",
    "segmenttree": "Range Queries",
    "segmenttrees": "Range Queries",
    "slidingwindow": "Two Pointers",
    "spraguegrundy": "Game Theory",
    "sqrtdecomposition": "Range Queries",
    "squarerootdecomposition": "Range Queries",
    "stringalgorithms": "String Algorithms",
    "stringalgos": "String Algorithms",
    "suffixarray": "String Algorithms",
    "suffixarrays": "String Algorithms",
    "twopointer": "Two Pointers",
    "twopointers": "Two Pointers",
    "unionfind": "Union Find",
    "zalgorithm": "String Algorithms",
}

LOW_SIGNAL_PATTERNS = [
    r"^(easy|veryeasy|basic|simple|medium|hard|easy medium|medium hard)$",
    r"^(special|restricted|completed|challenge|challenge problem|long challenge|monthly contest)$",
    r"^(algorithms|advanced algorithms|simple algos|algorithms warmup|fundamentals|tutorials|performance|properties|real life|base)$",
    r"^(cakewalk|cook off)$",
    r"^(admin\d*|.* adm|[a-z]+_adm)$",
    r"^(cook\d+|start\d+|ltime\d+|jan\d+|feb\d+|march\d+|april\d+|may\d+|june\d+|july\d+|aug\d+|sept\d+|oct\d+|nov\d+|dec\d+)$",
    r"^[a-z0-9]+_[a-z0-9]+$",
    r"^[a-z]+\d+[a-z0-9]*$",
]


def normalize_tag_text(label: object) -> tuple[str, str]:
    raw = str(label or "").strip()
    raw = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", raw)
    raw = raw.replace("&", " and ")
    text = raw.lower()
    text = re.sub(r"[_\-/]+", " ", text)
    text = re.sub(r"[^a-z0-9+ ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text, text.replace(" ", "")


def is_empty_original_domain(value: object) -> bool:
    return str(value if value is not None else "").strip() in EMPTY_ORIGINAL_DOMAIN_VALUES


def parse_list_cell(value: object) -> list[str]:
    raw = str(value if value is not None else "").strip()
    if not raw:
        return []
    try:
        parsed = ast.literal_eval(raw) if raw.startswith("[") else raw
    except (SyntaxError, ValueError):
        parsed = raw
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    return [str(parsed).strip()] if str(parsed).strip() else []


def canonicalize_existing_label(label: str) -> str:
    if label in CANONICAL_SET:
        return label
    return normalize_raw_tag(label)


def normalize_raw_tag(label: object) -> str:
    text, compact = normalize_tag_text(label)
    if not text:
        return LOW_SIGNAL_FALLBACK_LABEL
    if text in EXACT_ALIASES:
        return EXACT_ALIASES[text]
    if compact in COMPACT_ALIASES:
        return COMPACT_ALIASES[compact]

    if re.search(r"\b(0 1 bfs|dijkstra|floyd|warshall|bellman|shortest|sssp|all pairs)\b", text):
        return "Shortest Paths"
    if re.search(r"\b(segment tree|fenwick|binary indexed tree|range query|range queries|range minimum|range sum|prefix sum|suffix sum|sparse table|sqrt decomposition|square root decomposition|mo s algorithm|offline queries|online queries|queries|lazy propagation)\b", text):
        return "Range Queries"
    if re.search(r"\b(two pointer|two pointers|sliding window)\b", text):
        return "Two Pointers"
    if re.search(r"\b(divide and conquer|meet in the middle|meet in middle)\b", text):
        return "Divide and Conquer"

    if re.search(r"\b(dp|dynamic|memoization|memo|knapsack|digit dp|sos dp|lis|lcs|longest increasing subsequence|longest common substring|recurrence relation|state machines|bottom up|top down|fibonacci|kadane)\b", text):
        return "Dynamic Programming"
    if re.search(r"\b(lca|lowest common ancestor|hld|heavy light|euler tour|auxiliary tree|tree queries|tree algos|tree algorithms|binary tree|trees?\b|tree ds|dsu on trees)\b", text):
        return "Tree Algorithms"
    if re.search(r"\b(graph|graphs|dfs|bfs|depth first|breadth first|traversal|traversals|connected components|connectivity|reachability|bipartite|dag|directed acyclic|topological|topsort|scc|strong connectivity|strongly connected|spanning tree|mst|kruskal|prim|flow|flows|cut|max flow|min cut|matching|kuhn|hopcroft|2 sat|hamiltonian|vertex cover|planar graph|graph isomorphism|flood fill|cycle graph|cycles?|networks)\b", text):
        return "Graph Algorithms"

    if re.search(r"\b(union find|disjoint set|dsu)\b", text):
        return "Union Find"
    if re.search(r"\b(data structure|data structures|array|arrays|3d arrays|stacks?|queues?|deque|heap|priority queue|hash|hashing|hashmaps?|hash table|linked list|lists?|sets?|maps?|multiset|ordered set|unordered set|bst|treap|splay|avl|persistent structures|policy based|pbds|monotonic stack|monotonic queue|next greater element|coordinate compression|difference array|frequency array|amortized analysis|cyclic rotation|subarray|grid)\b", text):
        return "Data Structures"
    if re.search(r"\b(string|strings|substring|subsequence|suffix|trie|tries|kmp|z algorithm|aho corasick|rabin karp|rolling hash|manacher|palindrome|palindromes|anagram|regex|regular expression|regular expressions|parsing|parser|expression parsing|pattern searching|alphabets|formal language|ciphers|string manipulation|hamming distance)\b", text):
        return "String Algorithms"

    if re.search(r"\b(matrix|matrices|matrix exponentiation|matrix multiplication|determinant|hadamard matrix|kirchhoffs matrix tree theorem)\b", text):
        return "Matrix"
    if re.search(r"\b(number theory|prime|primes|primality|sieve|gcd|lcm|divisor|divisors|divisibility|factor|factorial|factorization|factorisation|modular|modulo|big integer|arbitrary precision|chinese remainder|crt|euler|fermat|mobius|primitive root|quadratic residue|discrete logarithm|carmichael|garners|poly divisible|zeckendorf|euclid|bezout|exponentiation|integer division|cryptography|rabin miller)\b", text):
        return "Number Theory"
    if re.search(r"\b(combinatorics|combinatorial|combination|combinations|permutation|permutations|counting|catalan|binomial|inclusion exclusion|pigeonhole|burnside|derangement|partitions|sperner|dilworth|generating functions|groupings|coverings)\b", text):
        return "Combinatorics"
    if re.search(r"\b(probability|probabilities|expected value|expectation|statistics|distribution|randomized)\b", text):
        return "Probability"
    if re.search(r"\b(geometry|geometric|computational geometry|convex hull|sweep line|line sweep|scanline|coordinate|cartesian|polygon|polygons|triangle|triangles|lines?|cross product|rotating caliper|physics|trigonometry|image processing)\b", text):
        return "Geometry"
    if re.search(r"\b(game theory|games?|nim|sprague|grundy|hackenbush|impartial game|chess)\b", text):
        return "Game Theory"

    if re.search(r"\b(greedy|observation|activity selection|scheduling|schedule|schedules|heuristics|contribution trick|case work|optimization problems|logical thinking|logical|proof|advanced greedy)\b", text):
        return "Greedy Algorithms"
    if re.search(r"\b(brute force|complete search|exhaustive|enumeration|pruning)\b", text):
        return "Brute Force"
    if re.search(r"\b(binary search|ternary search|searching|search|quickselect|binary search on answer)\b", text):
        return "Binary Search"
    if re.search(r"\b(sort|sorting|sortings|merge sort|counting sort|radix sort|bucket sort|lexicographic order|inversions)\b", text):
        return "Sorting"
    if re.search(r"\b(bit manipulation|bitmask|bitmasks|bitmasking|bitwise|bits?|binary|binary representation|gray code|xor|shift|bit magic|subset enumeration|subset)\b", text):
        return "Bit Manipulation"
    if re.search(r"\b(backtracking)\b", text):
        return "Backtracking"

    if re.search(r"\b(constructive|constructive algorithms|constructive algo|patterns?|pattern printing|printing patterns|ascii art|transformations)\b", text):
        return "Constructive Algorithms"
    if re.search(r"\b(simulation|ad hoc|adhoc|puzzle|puzzles|interactive|date time|time|riddles?|brainteaser)\b", text):
        return "Simulation"
    if re.search(r"\b(recursion|recursive)\b", text):
        return "Recursion"
    if re.search(r"\b(math|maths|mathematics|mathematical|arithmetic|algebra|linear algebra|calculus|differentiation|integration|inequality|inequalities|logic|numbers|number system|series|sequence|sequences|progression|ap|gp|logarithm|log exp pow|squares|square root algorithms|functional equations|quantifiers|polynomial|polynomials|fft|ntt|convolution|interpolation|lagrange|gaussian elimination|cauchy|sylvesters|discrete mathematics|discrete maths|group theory|linear programming|beatty sequence|fractions|division)\b", text):
        return "Math"
    if re.search(r"\b(implementation|basic programming|basic programming concepts|loops?|looping|conditional statements|operators|data types|functions|inbuilt functions|language features|iterators|oop|object oriented|cpp|java|python|numpy|collections|built ins|classes|functionals|closures|decorators|errors|exceptions|xml|json|security|error detection|debugging|refactoring|preprocessing|pre processing|precomputation|pre computation|filtering|reverse|pointers|unicode|esoteric languages|interpreters|functional programming|machine learning|data science|design pattern|object oriented programming|cpp control flow|java collections)\b", text):
        return "Implementation"

    if any(re.search(pattern, text) for pattern in LOW_SIGNAL_PATTERNS):
        return LOW_SIGNAL_FALLBACK_LABEL
    return LOW_SIGNAL_FALLBACK_LABEL


def normalize_original_domain(value: object) -> list[str]:
    labels = parse_list_cell(value)
    normalized = [normalize_raw_tag(label) for label in labels]
    return dedupe_canonical_labels(normalized)


def normalize_existing_llm_labels(value: object) -> list[str]:
    labels = parse_list_cell(value)
    normalized = [canonicalize_existing_label(label) for label in labels]
    return dedupe_canonical_labels(normalized)


def dedupe_canonical_labels(labels: Iterable[str]) -> list[str]:
    seen = set()
    result = []
    for label in labels:
        final_label = label if label in CANONICAL_SET else normalize_raw_tag(label)
        if final_label not in CANONICAL_SET:
            final_label = LOW_SIGNAL_FALLBACK_LABEL
        if final_label not in seen:
            seen.add(final_label)
            result.append(final_label)
    return result or [LOW_SIGNAL_FALLBACK_LABEL]


def labels_to_critic_categories(labels: Iterable[str]) -> list[str]:
    categories = {CRITIC_TAXONOMY_MAP[label] for label in labels}
    return [category for category in CRITIC_PRIORITY if category in categories]


def choose_main_critic(categories: list[str]) -> str:
    return categories[0] if categories else CRITIC_TAXONOMY_MAP[LOW_SIGNAL_FALLBACK_LABEL]


def str_list(values: Iterable[str]) -> str:
    return str(list(values))


def build_output_paths(output_dir: Path, suffix: str) -> dict[str, Path]:
    return {
        "labelled": output_dir / f"04_execution_ready_final_labels_{suffix}.csv",
        "normalized_distribution": output_dir / f"final_normalized_label_distribution_{suffix}.csv",
        "critic_distribution": output_dir / f"final_critic_distribution_{suffix}.csv",
        "tag_report": output_dir / f"final_tag_normalization_report_{suffix}.csv",
    }


def write_counter_csv(path: Path, key_name: str, counter: Counter[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([key_name, "count"])
        for key, count in counter.most_common():
            writer.writerow([key, count])


def write_tag_report(tag_distribution_csv: Path, output_path: Path) -> Counter[str]:
    tag_counter = Counter()
    with tag_distribution_csv.open("r", newline="", encoding="utf-8") as in_handle, output_path.open(
        "w", newline="", encoding="utf-8"
    ) as out_handle:
        reader = csv.DictReader(in_handle)
        writer = csv.DictWriter(
            out_handle,
            fieldnames=["tag", "source_count", "normalized_label", "critic_category"],
        )
        writer.writeheader()
        for row in reader:
            tag = row.get("tag", "")
            try:
                count = int(row.get("count", 0))
            except ValueError:
                count = 0
            normalized_label = normalize_raw_tag(tag)
            critic_category = CRITIC_TAXONOMY_MAP[normalized_label]
            writer.writerow(
                {
                    "tag": tag,
                    "source_count": count,
                    "normalized_label": normalized_label,
                    "critic_category": critic_category,
                }
            )
            tag_counter[normalized_label] += count
    return tag_counter


def process_pipeline(
    input_csv_path: Path | str = INPUT_CSV,
    output_dir: Path | str = OUTPUT_DIR,
    tag_distribution_csv: Path | str = TAG_DISTRIBUTION_CSV,
    suffix: str = "manual",
) -> dict[str, Path]:
    csv.field_size_limit(sys.maxsize)

    input_csv_path = Path(input_csv_path)
    output_dir = Path(output_dir)
    tag_distribution_csv = Path(tag_distribution_csv)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = build_output_paths(output_dir, suffix)

    normalized_counter = Counter()
    critic_counter = Counter()
    main_critic_counter = Counter()
    source_counter = Counter()

    with input_csv_path.open("r", newline="", encoding="utf-8") as in_handle, output_paths["labelled"].open(
        "w", newline="", encoding="utf-8"
    ) as out_handle:
        reader = csv.DictReader(in_handle)
        output_columns = list(reader.fieldnames or [])
        for column in ["normalized_labels", "critic_categories", "main_critic_category", "label_source"]:
            if column not in output_columns:
                output_columns.append(column)

        writer = csv.DictWriter(out_handle, fieldnames=output_columns)
        writer.writeheader()

        for row in reader:
            original_domain_empty = is_empty_original_domain(row.get("original_domain"))
            if original_domain_empty:
                normalized_labels = normalize_existing_llm_labels(row.get("normalized_labels"))
                label_source = "llm_preserved"
            else:
                normalized_labels = normalize_original_domain(row.get("original_domain"))
                label_source = "rule_based_original_domain"

            critic_categories = labels_to_critic_categories(normalized_labels)
            main_critic_category = choose_main_critic(critic_categories)

            row["normalized_labels"] = str_list(normalized_labels)
            row["critic_categories"] = str_list(critic_categories)
            row["main_critic_category"] = main_critic_category
            row["label_source"] = label_source
            writer.writerow({column: row.get(column, "") for column in output_columns})

            source_counter[label_source] += 1
            for label in normalized_labels:
                normalized_counter[label] += 1
            for critic_category in critic_categories:
                critic_counter[critic_category] += 1
            main_critic_counter[main_critic_category] += 1

    write_counter_csv(output_paths["normalized_distribution"], "normalized_label", normalized_counter)
    write_counter_csv(output_paths["critic_distribution"], "critic_category", critic_counter)
    tag_counter = write_tag_report(tag_distribution_csv, output_paths["tag_report"])

    print(f"Saved labelled CSV: {output_paths['labelled']}")
    print(f"Saved normalized distribution: {output_paths['normalized_distribution']}")
    print(f"Saved critic distribution: {output_paths['critic_distribution']}")
    print(f"Saved tag normalization report: {output_paths['tag_report']}")
    print(f"Row sources: {dict(source_counter)}")
    print(f"Main critic distribution: {dict(main_critic_counter)}")
    print(f"Tag report normalized labels covered: {len(tag_counter)}")
    return output_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create final normalized labels and critic groups.")
    parser.add_argument("--input-csv", type=Path, default=INPUT_CSV)
    parser.add_argument("--tag-distribution-csv", type=Path, default=TAG_DISTRIBUTION_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--suffix", default="manual")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_pipeline(
        input_csv_path=args.input_csv,
        output_dir=args.output_dir,
        tag_distribution_csv=args.tag_distribution_csv,
        suffix=args.suffix,
    )


if __name__ == "__main__":
    main()
