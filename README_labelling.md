# Stage 2: GPT-based Multi-label Problem Type Annotation

This document describes the implementation stage **after** the completion of:

```text
01_unified_raw.jsonl
02_algorithmic_filtered.jsonl
03_deduplicated.jsonl
04_execution_ready.jsonl
```

The goal of this stage is:

```text
1. Build a normalized canonical problem-type taxonomy
2. Map heterogeneous benchmark-native tags into normalized domains
3. Generate GPT-based multi-label annotations
4. Produce MoE-ready routing labels
```

This stage still does NOT train the MoE critic yet.

---

# 1. Objective

The purpose of this stage is to convert inconsistent benchmark-native metadata into a small, stable, reusable routing taxonomy for future MoE critic training.

Important distinction:

```text
original_domain
    = raw native benchmark tags

category
    = normalized canonical routing labels for MoE
```

Do NOT directly train on native benchmark tags because:

```text
APPS tags are inconsistent
CodeContests tags may be missing
TACO skill tags differ from Codeforces tags
LiveCodeBench metadata differs by source
```

Instead:

```text
native tags
    → normalized domains
    → GPT-assisted multi-label taxonomy
    → final category labels
```

---

# 2. Goal of Final Labels

The final labels should satisfy:

```text
small number of stable routing types
multi-label allowed
human interpretable
useful for MoE routing
not benchmark-specific
```

Example final routing labels:

```text
Dynamic_Programming
Graph
Greedy
Data_Structures
Math
String
Simulation
Binary_Search
Tree
Geometry
```

Recommended size:

```text
5 ~ 15 labels total
```

Do NOT create hundreds of sparse labels.

---

# 3. Repository Additions

Extend repository structure:

```text
algorithmic_coding_critic_data/
│
├── configs/
│   ├── labeling_config.yaml
│   └── taxonomy.yaml
│
├── data/
│   ├── processed/
│   │   ├── 04_execution_ready.jsonl
│   │   ├── 05_domain_normalized.jsonl
│   │   ├── 06_gpt_labeled.jsonl
│   │   └── 07_moe_ready.jsonl
│   │
│   └── reports/
│       ├── domain_normalization_report.json
│       ├── gpt_labeling_report.json
│       └── taxonomy_stats.json
│
├── prompts/
│   ├── normalize_domain.txt
│   └── classify_problem_type.txt
│
└── src/
    └── acc_data_pipeline/
        ├── labeling/
        │   ├── normalize_domains.py
        │   ├── taxonomy.py
        │   ├── litellm_client.py
        │   ├── label_with_llm.py
        │   ├── parse_labels.py
        │   ├── validate_labels.py
        │   └── moe_export.py
        │
        └── cli/
            ├── normalize_domains.py
            ├── label_gpt.py
            └── export_moe.py
```

---

# 4. Input CSV Schema

The previous stage should already export a lightweight CSV or JSONL schema:

```csv
problem,answer,test_cases,eval_spec,source,source_platform,original_domain,category
```

Current meaning:

| Column            | Meaning                |
| ----------------- | ---------------------- |
| `problem`         | full problem statement |
| `answer`          | reference solutions    |
| `test_cases`      | execution tests        |
| `eval_spec`       | evaluation metadata    |
| `source`          | benchmark source       |
| `source_platform` | original platform      |
| `original_domain` | native benchmark tags  |
| `category`        | currently empty        |

---

# 5. Stage 2 Pipeline Overview

Pipeline order:

```text
04_execution_ready.jsonl
        ↓
normalize native domains
        ↓
05_domain_normalized.jsonl
        ↓
LLM-based multi-label classification
        ↓
06_gpt_labeled.jsonl
        ↓
validation + cleaning
        ↓
07_moe_ready.jsonl
```

---

# 6. Domain Normalization

Implement:

```text
src/acc_data_pipeline/labeling/normalize_domains.py
```

Purpose:

```text
normalize heterogeneous native tags
before GPT labeling
```

Example:

| Original tag          | Normalized domain     |
| --------------------- | --------------------- |
| `dp`                  | `dynamic_programming` |
| `dynamic programming` | `dynamic_programming` |
| `graphs`              | `graph`               |
| `graph theory`        | `graph`               |
| `dfs and similar`     | `graph`               |
| `data structures`     | `data_structure`      |
| `implementation`      | `implementation`      |

---

# 7. Taxonomy Definition

Create:

```text
configs/taxonomy.yaml
```

Example:

```yaml
canonical_labels:
  - Dynamic_Programming
  - Graph
  - Greedy
  - Data_Structures
  - Math
  - String
  - Tree
  - Binary_Search
  - Simulation
  - Geometry

aliases:
  dp: Dynamic_Programming
  dynamic_programming: Dynamic_Programming
  graphs: Graph
  graph_theory: Graph
  greedy_algorithms: Greedy
```

Important:

```text
taxonomy labels must be stable
taxonomy labels must be low-cardinality
taxonomy labels should not depend on benchmark naming
```

---

# 8. Native Domain Normalization CLI

Implement:

```bash
python -m acc_data_pipeline.cli.normalize_domains \
  --input data/processed/04_execution_ready.jsonl \
  --taxonomy configs/taxonomy.yaml \
  --output data/processed/05_domain_normalized.jsonl \
  --report data/reports/domain_normalization_report.json
```

This stage should:

```text
read original_domain
normalize native tags
map aliases
remove duplicates
store normalized domains
preserve original native tags
```

---

# 9. Extended Record Schema

After normalization:

```json
{
  "problem": "...",
  "original_domain": [
    "dp",
    "graphs"
  ],
  "normalized_domains": [
    "Dynamic_Programming",
    "Graph"
  ],
  "category": []
}
```

Important:

```text
normalized_domains
    != final GPT labels

normalized_domains
    are only hints
```

---

# 10. LiteLLM Integration

Implement:

```text
src/acc_data_pipeline/labeling/litellm_client.py
```

Use LiteLLM as a unified inference layer.

Supported deployment modes:

```text
1. OpenAI-compatible API
2. vLLM OpenAI server
3. Local on-prem GPU endpoint
4. Azure OpenAI compatible endpoint
```

Do NOT hardcode provider-specific logic.

---

# 11. LiteLLM Configuration

Create:

```yaml
# configs/labeling_config.yaml

provider: openai_compatible

model: gpt-4.1-mini

api_base: http://localhost:8000/v1

api_key: EMPTY

temperature: 0.0

max_tokens: 128

batch_size: 32

multi_label: true

allowed_labels:
  - Dynamic_Programming
  - Graph
  - Greedy
  - Data_Structures
  - Math
  - String
  - Tree
  - Binary_Search
  - Simulation
  - Geometry
```

---

# 12. Supported Inference Backends

The implementation must support both:

## 12.1 Hosted API

Example:

```yaml
provider: openai
model: gpt-4.1-mini
api_key: ${OPENAI_API_KEY}
```

---

## 12.2 On-prem GPU via vLLM

Example:

```yaml
provider: openai_compatible

model: Qwen/Qwen3-32B

api_base: http://localhost:8000/v1

api_key: EMPTY
```

Example vLLM launch:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-32B \
  --tensor-parallel-size 4 \
  --port 8000
```

LiteLLM should call this endpoint using OpenAI-compatible API format.

---

# 13. LiteLLM Client Implementation

Implement:

```python
from litellm import completion

response = completion(
    model=model_name,
    api_base=api_base,
    api_key=api_key,
    messages=messages,
    temperature=0.0,
    max_tokens=128
)
```

Do NOT implement raw OpenAI SDK directly.

Always use LiteLLM abstraction.

---

# 14. Prompt Design

Create:

```text
prompts/classify_problem_type.txt
```

Prompt objective:

```text
Given a coding problem,
predict one or more canonical algorithm categories.
```

Important:

```text
allow multiple labels
do not force exactly one label
do not generate unseen labels
```

---

# 15. Prompt Template

Recommended prompt:

```text
You are classifying algorithmic programming problems.

Choose one or more labels from:

- Dynamic_Programming
- Graph
- Greedy
- Data_Structures
- Math
- String
- Tree
- Binary_Search
- Simulation
- Geometry

Rules:
- Multiple labels are allowed
- Return only labels
- Do not explain
- Do not invent new labels

Problem:
{problem_statement}

Normalized domain hints:
{normalized_domains}

Return JSON:

{
  "labels": [...]
}
```

---

# 16. Multi-label Rules

Important design rule:

```text
A problem may belong to multiple algorithmic types.
```

Examples:

| Problem                         | Labels                             |
| ------------------------------- | ---------------------------------- |
| shortest path + dp              | `["Graph", "Dynamic_Programming"]` |
| binary search on answer         | `["Binary_Search", "Greedy"]`      |
| segment tree + lazy propagation | `["Data_Structures", "Tree"]`      |

Do NOT collapse multi-domain problems into a single label.

---

# 17. Labeling CLI

Implement:

```bash
python -m acc_data_pipeline.cli.label_gpt \
  --input data/processed/05_domain_normalized.jsonl \
  --taxonomy configs/taxonomy.yaml \
  --config configs/labeling_config.yaml \
  --output data/processed/06_gpt_labeled.jsonl \
  --report data/reports/gpt_labeling_report.json
```

This command should:

```text
batch inference through LiteLLM
retry failed requests
validate returned labels
drop invalid labels
preserve raw LLM outputs
write multi-label categories
```

---

# 18. Output Schema

After labeling:

```json
{
  "problem_id": "...",
  "normalized_domains": [
    "Graph"
  ],
  "category": [
    "Graph",
    "Dynamic_Programming"
  ],
  "llm_annotation": {
    "model": "gpt-4.1-mini",
    "raw_response": "{...}",
    "timestamp": "...",
    "prompt_version": "v1"
  }
}
```

---

# 19. Label Validation

Implement:

```text
src/acc_data_pipeline/labeling/validate_labels.py
```

Validation rules:

```text
remove unknown labels
deduplicate labels
enforce allowed taxonomy
normalize capitalization
limit max labels per problem
```

Recommended:

```yaml
max_labels_per_problem: 3
```

---

# 20. Handling Missing Native Tags

Many problems may have:

```json
"original_domain": []
```

This is expected.

LLM classification should still work using:

```text
problem statement
examples
constraints
```

Do NOT skip unlabeled problems.

---

# 21. Recommended Sampling Strategy

For cost reduction:

```text
first classify only unique normalized statements
reuse labels for exact duplicates
cache prompts and responses
```

Recommended cache key:

```text
sha256(normalized_problem_statement)
```

---

# 22. Parallel Batch Inference

Implement batch inference carefully.

Recommended:

```yaml
batch_size: 32
max_concurrency: 8
retry_attempts: 3
```

Use async requests if possible.

---

# 23. Error Handling

Handle:

```text
timeout
invalid JSON
empty response
rate limit
unknown labels
hallucinated labels
```

Do NOT silently discard failures.

Store failures in report.

Example:

```json
{
  "problem_id": "...",
  "status": "failed",
  "reason": "invalid_json_response"
}
```

---

# 24. Final MoE-ready Export

Implement:

```bash
python -m acc_data_pipeline.cli.export_moe \
  --input data/processed/06_gpt_labeled.jsonl \
  --output data/processed/07_moe_ready.jsonl
```

The final file should contain:

```json
{
  "problem_id": "...",
  "problem": "...",
  "category": [
    "Graph",
    "Dynamic_Programming"
  ],
  "answer": [...],
  "test_cases": [...],
  "eval_spec": {...}
}
```

This file becomes the supervision source for:

```text
MoE router training
critic specialization
top-k expert routing
expert-balanced sampling
```

---

# 25. MoE Routing Interpretation

Later routing behavior:

```text
Problem:
    shortest path with state compression

Labels:
    ["Graph", "Dynamic_Programming"]

Possible MoE routing:
    top-2 experts
```

This is why multi-label annotation is necessary.

---

# 26. Acceptance Criteria

Implementation is complete when:

```text
native tags are normalized
taxonomy aliases work
LiteLLM backend works
OpenAI-compatible endpoint works
multi-label annotation works
invalid labels are filtered
all outputs are deterministic
final MoE-ready export exists
```

Final artifact:

```text
data/processed/07_moe_ready.jsonl
```

This file becomes the input for future:

```text
MoE router training
critic specialization
expert balancing
routing evaluation
```
