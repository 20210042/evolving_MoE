# Algorithmic Coding Critic Data Integration Pipeline

This repository implements the data integration pipeline for an Algorithmic Coding Critic project.

The immediate goal is **not** to train a critic model yet.  
The goal of this stage is to build a reliable, reproducible, execution-ready dataset from multiple coding benchmarks.

Target benchmarks:

- APPS
- CodeContests
- TACO
- LiveCodeBench

The pipeline must normalize heterogeneous benchmark formats into a shared schema, retain only algorithmic code-generation or self-repair tasks, remove duplicates, and expose a unified execution evaluation interface.

---

## 1. Scope

### In scope

Implement the following steps:

```text
Step 1. Dataset-specific loaders
Step 2. Common schema normalization
Step 3. Algorithmic coding task filtering
Step 4. Duplicate and near-duplicate removal
Step 5. Unified execution evaluation interface
```

### Out of scope for this stage

Do not implement these yet:

```text
GPT-based problem type labeling
MoE router training
Critic target generation
Critic model training
Repair model training
LLM-based critique generation
```

This stage should produce clean, deduplicated, execution-ready problem records.

---

## 2. Design Goal

The core design principle is:

> Do not simply concatenate APPS, CodeContests, TACO, and LiveCodeBench.  
> Normalize them into a common schema while preserving source-specific metadata and execution semantics.

Different benchmarks may differ in:

```text
problem statement format
input/output representation
test case structure
function-call vs stdin/stdout evaluation
availability of native tags
difficulty labels
split names
source URLs
solution formats
hidden/public test availability
task variants
special judge requirements
```

The pipeline must preserve these differences explicitly in the normalized records.

---

## 3. Expected Repository Structure

Implement the repository with the following structure.

```text
algorithmic_coding_critic_data/
│
├── README.md
├── pyproject.toml
├── configs/
│   ├── dataset_config.yaml
│   ├── filter_config.yaml
│   ├── dedup_config.yaml
│   └── execution_config.yaml
│
├── data/
│   ├── raw/
│   │   ├── apps/
│   │   ├── codecontests/
│   │   ├── taco/
│   │   └── livecodebench/
│   │
│   ├── processed/
│   │   ├── 01_unified_raw.jsonl
│   │   ├── 02_algorithmic_filtered.jsonl
│   │   ├── 03_deduplicated.jsonl
│   │   └── 04_execution_ready.jsonl
│   │
│   └── reports/
│       ├── load_report.json
│       ├── filter_report.json
│       ├── dedup_report.json
│       ├── schema_validation_report.json
│       └── eval_mode_report.json
│
├── src/
│   └── acc_data_pipeline/
│       ├── __init__.py
│       │
│       ├── schemas/
│       │   ├── problem.py
│       │   ├── testcase.py
│       │   ├── execution.py
│       │   └── validation.py
│       │
│       ├── loaders/
│       │   ├── base.py
│       │   ├── apps_loader.py
│       │   ├── codecontests_loader.py
│       │   ├── taco_loader.py
│       │   └── livecodebench_loader.py
│       │
│       ├── preprocessing/
│       │   ├── normalize.py
│       │   ├── filter_algorithmic.py
│       │   ├── deduplicate.py
│       │   └── split_utils.py
│       │
│       ├── execution/
│       │   ├── base.py
│       │   ├── stdin_stdout_runner.py
│       │   ├── function_call_runner.py
│       │   ├── comparator.py
│       │   ├── sandbox.py
│       │   └── execution_interface.py
│       │
│       ├── reports/
│       │   ├── stats.py
│       │   └── write_reports.py
│       │
│       └── cli/
│           ├── load.py
│           ├── filter.py
│           ├── dedup.py
│           ├── prepare_execution.py
│           └── validate.py
│
└── tests/
    ├── test_schema.py
    ├── test_loaders.py
    ├── test_filter_algorithmic.py
    ├── test_deduplicate.py
    ├── test_execution_interface.py
    └── fixtures/
```

---

## 4. Pipeline Overview

The pipeline should run in this order.

```text
Raw benchmark files
        ↓
Dataset-specific loaders
        ↓
01_unified_raw.jsonl
        ↓
Algorithmic task filter
        ↓
02_algorithmic_filtered.jsonl
        ↓
Deduplication
        ↓
03_deduplicated.jsonl
        ↓
Execution interface preparation
        ↓
04_execution_ready.jsonl
```

Each stage must be reproducible and must write a report.

---

## 5. Command-line Interface

Implement the following CLI commands.

### 5.1 Load and normalize raw benchmarks

```bash
python -m acc_data_pipeline.cli.load \
  --raw-root data/raw \
  --datasets apps codecontests taco livecodebench \
  --output data/processed/01_unified_raw.jsonl \
  --report data/reports/load_report.json
```

This command should:

```text
load each benchmark
convert each record to the common schema
preserve source-specific native metadata
assign stable global problem IDs
write JSONL output
write loading statistics
```

---

### 5.2 Filter algorithmic coding tasks

```bash
python -m acc_data_pipeline.cli.filter \
  --input data/processed/01_unified_raw.jsonl \
  --output data/processed/02_algorithmic_filtered.jsonl \
  --config configs/filter_config.yaml \
  --report data/reports/filter_report.json
```

This command should:

```text
keep algorithmic code-generation tasks
keep self-repair tasks only when they include a full candidate solution and executable tests
exclude pure code execution tasks
exclude test output prediction tasks
exclude non-algorithmic software engineering tasks
exclude malformed records without executable evaluation information
```

---

### 5.3 Deduplicate problems

```bash
python -m acc_data_pipeline.cli.dedup \
  --input data/processed/02_algorithmic_filtered.jsonl \
  --output data/processed/03_deduplicated.jsonl \
  --config configs/dedup_config.yaml \
  --report data/reports/dedup_report.json
```

This command should:

```text
remove exact duplicates
detect near-duplicate problem statements
preserve source provenance
assign dedup_group_id
choose one canonical representative per duplicate group
write duplicate statistics
```

---

### 5.4 Prepare execution-ready records

```bash
python -m acc_data_pipeline.cli.prepare_execution \
  --input data/processed/03_deduplicated.jsonl \
  --output data/processed/04_execution_ready.jsonl \
  --config configs/execution_config.yaml \
  --report data/reports/eval_mode_report.json
```

This command should:

```text
validate test cases
infer or verify eval_mode
normalize input/output cases
attach execution_spec
mark unsupported execution cases explicitly
write execution-mode statistics
```

---

### 5.5 Validate final schema

```bash
python -m acc_data_pipeline.cli.validate \
  --input data/processed/04_execution_ready.jsonl \
  --report data/reports/schema_validation_report.json
```

This command should:

```text
validate every record against the Pydantic schema
report invalid records
report missing required fields
report unsupported eval modes
report empty statements or empty test cases
```

---

## 6. Common Problem Schema

All benchmark records must be converted into the following normalized structure.

Use Pydantic models to enforce this schema.

### 6.1 Top-level schema

```json
{
  "schema_version": "0.1",
  "problem_id": "apps__train__000001",
  "source": "APPS",
  "source_problem_id": "000001",
  "source_url": null,
  "source_platform": null,
  "split": "train",

  "task_family": "algorithmic_code_generation",

  "title": null,
  "problem_statement": {
    "raw": "...",
    "description": "...",
    "input_format": "...",
    "output_format": "...",
    "constraints": "...",
    "notes": null
  },

  "examples": [
    {
      "input": "...",
      "output": "...",
      "explanation": null
    }
  ],

  "test_cases": [
    {
      "case_id": "apps__train__000001__public__0000",
      "visibility": "public",
      "input": {
        "kind": "stdin",
        "value": "..."
      },
      "expected_output": {
        "kind": "stdout",
        "value": "..."
      },
      "metadata": {}
    }
  ],

  "reference_solutions": [
    {
      "solution_id": "apps__train__000001__ref__0000",
      "language": "python",
      "code": "...",
      "is_known_correct": true,
      "source": "benchmark",
      "metadata": {}
    }
  ],

  "starter_code": null,

  "eval_spec": {
    "eval_mode": "stdin_stdout",
    "language": "python",
    "entry_point": null,
    "comparison": {
      "type": "exact_or_token_match",
      "numeric_tolerance": null,
      "case_sensitive": true,
      "strip_trailing_whitespace": true
    },
    "timeout_seconds": 5,
    "memory_limit_mb": 512,
    "requires_special_judge": false,
    "unsupported_reason": null
  },

  "native_metadata": {
    "difficulty": null,
    "tags": [],
    "original_task_type": null,
    "raw_fields": {}
  },

  "dedup": {
    "normalized_statement_hash": null,
    "canonical_problem_id": null,
    "dedup_group_id": null,
    "is_duplicate": false,
    "duplicate_of": null,
    "dedup_method": null
  },

  "quality_flags": {
    "has_statement": true,
    "has_tests": true,
    "has_reference_solution": true,
    "has_native_tags": false,
    "is_algorithmic": true,
    "is_supported_for_execution": true,
    "requires_manual_review": false,
    "warnings": []
  }
}
```

---

## 7. Field Definitions

### 7.1 `problem_id`

Must be globally unique and deterministic.

Format:

```text
{source_lower}__{split}__{source_problem_id}
```

Examples:

```text
apps__train__000001
codecontests__valid__123456
taco__test__000999
livecodebench__v5__leetcode_abc123
```

Do not use random UUIDs unless the benchmark has no stable ID.  
If random fallback is unavoidable, also store a deterministic hash in `native_metadata`.

---

### 7.2 `source`

Allowed values:

```text
APPS
CodeContests
TACO
LiveCodeBench
```

---

### 7.3 `task_family`

Allowed values:

```text
algorithmic_code_generation
self_repair
pure_code_execution
test_output_prediction
non_algorithmic_code_task
unknown
```

For this stage, keep only:

```text
algorithmic_code_generation
self_repair
```

after filtering.

---

### 7.4 `problem_statement`

Do not merge everything into a single unstructured string only.

Store both:

```text
raw
description
input_format
output_format
constraints
notes
```

If the source benchmark only provides one raw statement, put it in both:

```text
problem_statement.raw
problem_statement.description
```

and leave the other fields as `null`.

---

### 7.5 `examples`

Examples are not necessarily the same as hidden test cases.  
Keep them separate from `test_cases` when the benchmark distinguishes them.

Each example should be:

```json
{
  "input": "...",
  "output": "...",
  "explanation": null
}
```

---

### 7.6 `test_cases`

Test cases must be normalized into a list.

Each test case must have:

```json
{
  "case_id": "...",
  "visibility": "public",
  "input": {
    "kind": "stdin",
    "value": "..."
  },
  "expected_output": {
    "kind": "stdout",
    "value": "..."
  },
  "metadata": {}
}
```

Allowed `visibility` values:

```text
public
hidden
generated
unknown
```

Allowed input kinds:

```text
stdin
function_args
raw
unknown
```

Allowed expected output kinds:

```text
stdout
return_value
raw
unknown
```

For stdin/stdout problems:

```json
"input": {
  "kind": "stdin",
  "value": "1 2\n"
},
"expected_output": {
  "kind": "stdout",
  "value": "3\n"
}
```

For function-call problems:

```json
"input": {
  "kind": "function_args",
  "value": {
    "args": [[1, 2, 3]],
    "kwargs": {}
  }
},
"expected_output": {
  "kind": "return_value",
  "value": 6
}
```

---

### 7.7 `reference_solutions`

Reference solutions should be preserved if available.

```json
{
  "solution_id": "...",
  "language": "python",
  "code": "...",
  "is_known_correct": true,
  "source": "benchmark",
  "metadata": {}
}
```

Allowed `source` values:

```text
benchmark
human
llm
synthetic
unknown
```

This stage should not generate new LLM solutions.  
Only preserve existing benchmark solutions.

---

### 7.8 `eval_spec`

The execution specification defines how a candidate solution should be evaluated.

```json
{
  "eval_mode": "stdin_stdout",
  "language": "python",
  "entry_point": null,
  "comparison": {
    "type": "exact_or_token_match",
    "numeric_tolerance": null,
    "case_sensitive": true,
    "strip_trailing_whitespace": true
  },
  "timeout_seconds": 5,
  "memory_limit_mb": 512,
  "requires_special_judge": false,
  "unsupported_reason": null
}
```

Allowed `eval_mode` values:

```text
stdin_stdout
function_call
special_judge
self_repair
unsupported
```

Use `unsupported` when the record cannot be executed safely or consistently.

Do not drop unsupported records silently.  
Mark them explicitly and set:

```json
"quality_flags": {
  "is_supported_for_execution": false,
  "requires_manual_review": true
}
```

---

## 8. Benchmark-specific Loading Rules

### 8.1 APPS Loader

Implement:

```text
src/acc_data_pipeline/loaders/apps_loader.py
```

The APPS loader should:

```text
load problem statement
load input/output test information
load starter code if available
load reference solutions if available
detect function-call mode when function name is provided
otherwise use stdin/stdout mode
preserve difficulty if available
preserve all source-specific fields in native_metadata.raw_fields
```

Expected mapping:

```text
APPS question/description -> problem_statement.raw
APPS input_output -> test_cases
APPS solutions -> reference_solutions
APPS fn_name, if present -> eval_spec.entry_point
APPS difficulty -> native_metadata.difficulty
```

Evaluation mode rule:

```text
if fn_name exists:
    eval_mode = function_call
else:
    eval_mode = stdin_stdout
```

---

### 8.2 CodeContests Loader

Implement:

```text
src/acc_data_pipeline/loaders/codecontests_loader.py
```

The CodeContests loader should:

```text
load problem description
load source platform if available
load source URL if available
load public/private/generated tests if available
load correct solutions as reference_solutions
preserve incorrect solutions only as native metadata for now
preserve all source-specific fields in native_metadata.raw_fields
```

Expected mapping:

```text
description -> problem_statement.raw
public/private/generated tests -> test_cases with visibility field
correct solutions -> reference_solutions
incorrect solutions -> native_metadata.raw_fields.incorrect_solutions
source/platform -> source_platform
url -> source_url
```

Evaluation mode rule:

```text
default eval_mode = stdin_stdout
if special judge metadata exists:
    eval_mode = special_judge
```

Do not train or evaluate incorrect solutions in this stage.  
Just preserve them if available.

---

### 8.3 TACO Loader

Implement:

```text
src/acc_data_pipeline/loaders/taco_loader.py
```

The TACO loader should:

```text
load problem statement
load input/output examples and tests
load solutions if available
load difficulty if available
load native topic, skill, and algorithm tags if available
preserve source benchmark information
preserve all source-specific fields in native_metadata.raw_fields
```

Expected mapping:

```text
question/statement -> problem_statement.raw
input/output -> test_cases
solutions -> reference_solutions
topic/tags/skills -> native_metadata.tags
difficulty -> native_metadata.difficulty
```

Evaluation mode rule:

```text
if function-call style metadata exists:
    eval_mode = function_call
elif special judge metadata exists:
    eval_mode = special_judge
else:
    eval_mode = stdin_stdout
```

TACO may overlap with APPS and CodeContests.  
The deduplication stage must handle this.

---

### 8.4 LiveCodeBench Loader

Implement:

```text
src/acc_data_pipeline/loaders/livecodebench_loader.py
```

The LiveCodeBench loader must be careful because LiveCodeBench can include multiple task variants.

Allowed task variants for this stage:

```text
code_generation
self_repair
```

Excluded task variants for this stage:

```text
code_execution
test_output_prediction
other non-code-generation tasks
```

Expected mapping:

```text
problem statement -> problem_statement.raw
public/private tests -> test_cases
task variant -> native_metadata.original_task_type
platform/source -> source_platform
url -> source_url
```

Evaluation mode rule:

```text
if task variant is code_generation:
    eval_mode = stdin_stdout or function_call depending on test structure

if task variant is self_repair:
    eval_mode = self_repair

if task variant is code_execution or test_output_prediction:
    task_family should be set accordingly
    these records should be removed by the filtering stage
```

Do not silently convert output prediction tasks into code-generation tasks.

---

## 9. Algorithmic Task Filtering

Implement:

```text
src/acc_data_pipeline/preprocessing/filter_algorithmic.py
```

The filtering stage should decide whether a normalized record is suitable for this project.

Keep records when:

```text
task_family is algorithmic_code_generation or self_repair
problem_statement.raw is non-empty
at least one executable test case exists
eval_spec.eval_mode is not unsupported
the task requires producing or repairing a full program/function solution
```

Remove records when:

```text
task_family is pure_code_execution
task_family is test_output_prediction
task_family is non_algorithmic_code_task
problem statement is missing
test cases are missing
evaluation mode is unsupported and cannot be normalized
the task is only about predicting output
the task is only about explaining code behavior
the task is general software engineering rather than algorithmic programming
```

The filter must write removal reasons.

Example removal report item:

```json
{
  "problem_id": "livecodebench__v5__abc123",
  "source": "LiveCodeBench",
  "removed": true,
  "reason": "test_output_prediction_task"
}
```

---

## 10. Deduplication

Implement:

```text
src/acc_data_pipeline/preprocessing/deduplicate.py
```

The deduplication stage must remove duplicate and near-duplicate problems across benchmarks.

### 10.1 Normalization for deduplication

Implement a text normalization function:

```text
lowercase
strip whitespace
remove excessive newlines
normalize punctuation spacing
remove source-specific boilerplate if obvious
normalize numbers only if safe
do not remove core problem constraints
```

Generate:

```text
normalized_statement
normalized_statement_hash
```

Use SHA-256 for exact normalized hashes.

---

### 10.2 Exact duplicate detection

Exact duplicates should be detected using:

```text
normalized problem statement hash
source URL
title + normalized statement hash
identical example input/output pairs
```

---

### 10.3 Near-duplicate detection

Implement near-duplicate detection using at least one of:

```text
MinHash
character n-gram Jaccard similarity
token n-gram Jaccard similarity
TF-IDF cosine similarity
```

Recommended default:

```text
token 5-gram Jaccard similarity >= 0.85
```

The threshold should be configurable in:

```text
configs/dedup_config.yaml
```

Example:

```yaml
exact_hash: true
near_duplicate:
  method: "token_ngram_jaccard"
  ngram_size: 5
  threshold: 0.85
```

---

### 10.4 Canonical representative selection

When duplicates are found, choose one canonical representative.

Preferred source priority:

```text
LiveCodeBench
TACO
CodeContests
APPS
```

Reason:

```text
LiveCodeBench is often more recent
TACO may contain richer metadata
CodeContests may contain strong competitive programming tests
APPS is useful but may overlap with others
```

If source priority ties, choose the record with:

```text
more test cases
more reference solutions
more complete statement fields
native tags available
source URL available
```

Store dedup metadata:

```json
"dedup": {
  "normalized_statement_hash": "...",
  "canonical_problem_id": "taco__train__000123",
  "dedup_group_id": "dedup_group_000042",
  "is_duplicate": false,
  "duplicate_of": null,
  "dedup_method": "near_duplicate_token_5gram_jaccard"
}
```

For removed duplicate records, write them to the dedup report rather than losing all provenance.

---

## 11. Unified Execution Interface

Implement:

```text
src/acc_data_pipeline/execution/
```

This stage should define a common way to evaluate candidate code later.

It does not need to generate candidate solutions yet.  
However, it must prepare records so that future candidate solutions can be executed consistently.

---

### 11.1 Execution interface API

Implement a base interface:

```python
class ExecutionRunner:
    def can_run(self, problem: NormalizedProblem) -> bool:
        ...

    def run(
        self,
        problem: NormalizedProblem,
        candidate_code: str,
        language: str = "python"
    ) -> ExecutionResult:
        ...
```

Implement runners:

```text
StdinStdoutRunner
FunctionCallRunner
SpecialJudgeRunner
```

For `self_repair`, the final repaired code should eventually be evaluated using either:

```text
stdin_stdout
function_call
special_judge
```

So for this stage, `self_repair` should be represented as a task family, not as a completely separate execution mechanism unless the benchmark requires it.

---

### 11.2 Execution result schema

Implement a standard execution result.

```json
{
  "problem_id": "apps__train__000001",
  "solution_id": "candidate_000001",
  "passed": false,
  "status": "wrong_answer",
  "num_tests_passed": 7,
  "num_tests_total": 12,
  "failed_tests": [
    {
      "case_id": "apps__train__000001__hidden__0007",
      "input": {
        "kind": "stdin",
        "value": "..."
      },
      "expected_output": {
        "kind": "stdout",
        "value": "..."
      },
      "actual_output": {
        "kind": "stdout",
        "value": "..."
      },
      "error_message": null
    }
  ],
  "runtime_seconds": 0.23,
  "memory_mb": null,
  "stderr": null
}
```

Allowed status values:

```text
accepted
wrong_answer
runtime_error
timeout
compile_error
memory_limit_exceeded
unsupported_language
unsupported_eval_mode
internal_error
```

---

### 11.3 Comparator

Implement:

```text
src/acc_data_pipeline/execution/comparator.py
```

Supported comparison types:

```text
exact
exact_or_token_match
numeric_tolerance
case_insensitive
special_judge_unresolved
```

Default comparison:

```text
exact_or_token_match
```

Comparison behavior:

```text
strip trailing whitespace by default
normalize final newline
for token match, split by whitespace and compare token sequence
for numeric tolerance, compare floats with configured absolute/relative tolerance
if special judge is required but not implemented, mark as unsupported or requires_manual_review
```

---

### 11.4 Sandbox

Implement:

```text
src/acc_data_pipeline/execution/sandbox.py
```

All code execution must be sandboxed.

Minimum requirements:

```text
run in a temporary directory
enforce timeout
capture stdout
capture stderr
do not allow network access if possible
do not write outside temporary directory
clean up after execution
```

For the initial implementation, Python-only execution is acceptable.

Use subprocess with timeout.

Default limits:

```yaml
timeout_seconds: 5
memory_limit_mb: 512
max_output_bytes: 200000
```

If safe memory limiting is not implemented on the current OS, record this limitation in the report.

Do not execute arbitrary code without timeout.

---

## 12. Configuration Files

### 12.1 `configs/filter_config.yaml`

```yaml
allowed_task_families:
  - algorithmic_code_generation
  - self_repair

excluded_task_families:
  - pure_code_execution
  - test_output_prediction
  - non_algorithmic_code_task

require_problem_statement: true
require_test_cases: true
require_supported_eval_mode: true

min_statement_chars: 30
```

---

### 12.2 `configs/dedup_config.yaml`

```yaml
source_priority:
  - LiveCodeBench
  - TACO
  - CodeContests
  - APPS

exact:
  use_statement_hash: true
  use_source_url: true
  use_examples: true

near_duplicate:
  enabled: true
  method: token_ngram_jaccard
  ngram_size: 5
  threshold: 0.85

canonical_selection:
  prefer_more_test_cases: true
  prefer_reference_solution: true
  prefer_native_tags: true
  prefer_source_url: true
```

---

### 12.3 `configs/execution_config.yaml`

```yaml
default_language: python

supported_languages:
  - python

default_timeout_seconds: 5
default_memory_limit_mb: 512
max_output_bytes: 200000

comparison:
  default_type: exact_or_token_match
  strip_trailing_whitespace: true
  normalize_final_newline: true
  case_sensitive: true
  numeric_tolerance:
    abs_tol: 1.0e-6
    rel_tol: 1.0e-6

unsupported:
  keep_records: true
  mark_requires_manual_review: true
```

---

## 13. Reports

Each stage must write a report.

### 13.1 Load report

```json
{
  "total_loaded": 0,
  "by_source": {
    "APPS": 0,
    "CodeContests": 0,
    "TACO": 0,
    "LiveCodeBench": 0
  },
  "schema_errors": [],
  "warnings": []
}
```

---

### 13.2 Filter report

```json
{
  "input_count": 0,
  "kept_count": 0,
  "removed_count": 0,
  "removed_by_reason": {
    "test_output_prediction_task": 0,
    "missing_tests": 0,
    "missing_statement": 0,
    "unsupported_eval_mode": 0
  }
}
```

---

### 13.3 Dedup report

```json
{
  "input_count": 0,
  "output_count": 0,
  "num_duplicate_groups": 0,
  "num_removed_duplicates": 0,
  "duplicates_by_source_pair": {},
  "near_duplicate_threshold": 0.85
}
```

---

### 13.4 Execution mode report

```json
{
  "input_count": 0,
  "supported_count": 0,
  "unsupported_count": 0,
  "by_eval_mode": {
    "stdin_stdout": 0,
    "function_call": 0,
    "special_judge": 0,
    "unsupported": 0
  },
  "requires_manual_review": 0
}
```

---

## 14. Testing Requirements

Implement unit tests for:

```text
schema validation
each dataset loader with fixture data
algorithmic filtering
exact deduplication
near-duplicate deduplication
stdin/stdout execution runner
function-call execution runner
comparator behavior
unsupported eval mode handling
```

Minimum tests:

```text
tests/test_schema.py
tests/test_loaders.py
tests/test_filter_algorithmic.py
tests/test_deduplicate.py
tests/test_execution_interface.py
```

Use small synthetic fixtures.  
Do not require downloading the full benchmarks to run unit tests.

---

## 15. Acceptance Criteria

The implementation is complete when the following conditions are met.

### 15.1 Data loading

```text
All four benchmark loaders exist.
Each loader outputs records conforming to the common schema.
Source-specific metadata is preserved.
Problem IDs are deterministic.
```

### 15.2 Filtering

```text
Algorithmic code-generation tasks are kept.
Self-repair tasks with executable tests are kept.
Code execution and test output prediction tasks are excluded.
Every removed record has a removal reason.
```

### 15.3 Deduplication

```text
Exact duplicate detection works.
Near-duplicate detection works.
Duplicate groups are reported.
A canonical representative is selected deterministically.
Duplicate provenance is preserved in the report.
```

### 15.4 Execution interface

```text
stdin_stdout eval mode is supported.
function_call eval mode is supported.
special_judge cases are marked explicitly if not executable.
ExecutionResult schema is implemented.
Timeout is enforced.
stdout and stderr are captured.
Unsupported cases are not silently dropped.
```

### 15.5 Final output

The final output file must exist:

```text
data/processed/04_execution_ready.jsonl
```

Every record in this file must have:

```text
problem_id
source
task_family
problem_statement.raw
test_cases
eval_spec
native_metadata
quality_flags
dedup
```

---

## 16. Important Implementation Notes

### 16.1 Do not over-normalize

Do not destroy benchmark-specific information.

For example, do not convert every task into a plain stdin/stdout task if the original task is function-call based.

Preserve the actual evaluation semantics using:

```text
eval_spec.eval_mode
test_cases.input.kind
test_cases.expected_output.kind
eval_spec.entry_point
```

---

### 16.2 Do not trust native tags

Native tags should be stored but not used as canonical labels.

This stage should not create the final 5-type problem taxonomy.

Store native tags only here:

```json
"native_metadata": {
  "tags": []
}
```

Canonical tags will be generated later by GPT-based annotation.

---

### 16.3 Do not silently discard difficult cases

If a record has special judge behavior, malformed tests, or unsupported language, keep it only if configured to do so, but mark it clearly.

Use:

```json
"quality_flags": {
  "is_supported_for_execution": false,
  "requires_manual_review": true,
  "warnings": ["special_judge_not_implemented"]
}
```

---

### 16.4 Keep raw provenance

Every normalized record should retain enough source metadata to trace it back.

At minimum:

```text
source
source_problem_id
source_url if available
source_platform if available
split
native_metadata.raw_fields
```

Do not store extremely large raw blobs if they make JSONL unusable.  
If needed, store a path or compact subset.

---

## 17. Recommended Implementation Order

Implement in this order:

```text
1. Pydantic schemas
2. JSONL read/write utilities
3. APPS loader
4. TACO loader
5. CodeContests loader
6. LiveCodeBench loader
7. Schema validation CLI
8. Algorithmic filtering
9. Deduplication
10. Execution spec preparation
11. Comparator
12. StdinStdoutRunner
13. FunctionCallRunner
14. Execution interface tests
15. Reports
```

Do not start MoE or GPT labeling code before this pipeline is stable.

---

## 18. Final Deliverable

At the end of this stage, the repository should produce:

```text
data/processed/04_execution_ready.jsonl
```

This file is the input to the next stage:

```text
GPT-based 5-type multi-label annotation
MoE router supervision
Algorithmic Coding Critic training
```

The current stage must stop before those steps.
