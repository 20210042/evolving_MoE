from __future__ import annotations

from copy import deepcopy
from typing import Any, ClassVar, Literal

try:
    from pydantic import BaseModel, ConfigDict, Field

    PYDANTIC_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal envs
    BaseModel = object  # type: ignore
    ConfigDict = dict  # type: ignore
    Field = None  # type: ignore
    PYDANTIC_AVAILABLE = False


SOURCE_VALUES = {"APPS", "CodeContests", "TACO", "LiveCodeBench"}
TASK_FAMILY_VALUES = {
    "algorithmic_code_generation",
    "self_repair",
    "pure_code_execution",
    "test_output_prediction",
    "non_algorithmic_code_task",
    "unknown",
}
VISIBILITY_VALUES = {"public", "hidden", "generated", "unknown"}
INPUT_KIND_VALUES = {"stdin", "function_args", "raw", "unknown"}
OUTPUT_KIND_VALUES = {"stdout", "return_value", "raw", "unknown"}
EVAL_MODE_VALUES = {
    "stdin_stdout",
    "function_call",
    "special_judge",
    "self_repair",
    "unsupported",
}
COMPARISON_VALUES = {
    "exact",
    "exact_or_token_match",
    "numeric_tolerance",
    "case_insensitive",
    "special_judge_unresolved",
}
REFERENCE_SOURCE_VALUES = {"benchmark", "human", "llm", "synthetic", "unknown"}


def _field(default: Any = None, default_factory: Any = None) -> Any:
    if PYDANTIC_AVAILABLE:
        if default_factory is not None:
            return Field(default_factory=default_factory)
        return Field(default=default)
    if default_factory is not None:
        return default_factory()
    return default


if PYDANTIC_AVAILABLE:

    class ACCBaseModel(BaseModel):
        model_config = ConfigDict(extra="allow")


    class ProblemStatement(ACCBaseModel):
        raw: str
        description: str | None = None
        input_format: str | None = None
        output_format: str | None = None
        constraints: str | None = None
        notes: str | None = None


    class Example(ACCBaseModel):
        input: Any
        output: Any
        explanation: str | None = None


    class IOValue(ACCBaseModel):
        kind: Literal[
            "stdin",
            "function_args",
            "stdout",
            "return_value",
            "raw",
            "unknown",
        ]
        value: Any


    class TestCase(ACCBaseModel):
        case_id: str
        visibility: Literal["public", "hidden", "generated", "unknown"]
        input: IOValue
        expected_output: IOValue
        metadata: dict[str, Any] = Field(default_factory=dict)


    class ReferenceSolution(ACCBaseModel):
        solution_id: str
        language: str = "python"
        code: str
        is_known_correct: bool = True
        source: Literal["benchmark", "human", "llm", "synthetic", "unknown"] = "benchmark"
        metadata: dict[str, Any] = Field(default_factory=dict)


    class ComparisonSpec(ACCBaseModel):
        type: Literal[
            "exact",
            "exact_or_token_match",
            "numeric_tolerance",
            "case_insensitive",
            "special_judge_unresolved",
        ] = "exact_or_token_match"
        numeric_tolerance: dict[str, float] | None = None
        case_sensitive: bool = True
        strip_trailing_whitespace: bool = True


    class EvalSpec(ACCBaseModel):
        eval_mode: Literal[
            "stdin_stdout",
            "function_call",
            "special_judge",
            "self_repair",
            "unsupported",
        ] = "stdin_stdout"
        language: str = "python"
        entry_point: str | None = None
        comparison: ComparisonSpec = Field(default_factory=ComparisonSpec)
        timeout_seconds: float = 5
        memory_limit_mb: int | None = 512
        requires_special_judge: bool = False
        unsupported_reason: str | None = None


    class NativeMetadata(ACCBaseModel):
        difficulty: str | None = None
        tags: list[str] = Field(default_factory=list)
        original_task_type: str | None = None
        raw_fields: dict[str, Any] = Field(default_factory=dict)


    class DedupMetadata(ACCBaseModel):
        normalized_statement_hash: str | None = None
        canonical_problem_id: str | None = None
        dedup_group_id: str | None = None
        is_duplicate: bool = False
        duplicate_of: str | None = None
        dedup_method: str | None = None


    class QualityFlags(ACCBaseModel):
        has_statement: bool = True
        has_tests: bool = True
        has_reference_solution: bool = False
        has_native_tags: bool = False
        is_algorithmic: bool = True
        is_supported_for_execution: bool = True
        requires_manual_review: bool = False
        warnings: list[str] = Field(default_factory=list)


    class NormalizedProblem(ACCBaseModel):
        schema_version: str = "0.1"
        problem_id: str
        source: Literal["APPS", "CodeContests", "TACO", "LiveCodeBench"]
        source_problem_id: str
        source_url: str | None = None
        source_platform: str | None = None
        split: str
        task_family: Literal[
            "algorithmic_code_generation",
            "self_repair",
            "pure_code_execution",
            "test_output_prediction",
            "non_algorithmic_code_task",
            "unknown",
        ] = "algorithmic_code_generation"
        title: str | None = None
        problem_statement: ProblemStatement
        examples: list[Example] = Field(default_factory=list)
        test_cases: list[TestCase] = Field(default_factory=list)
        reference_solutions: list[ReferenceSolution] = Field(default_factory=list)
        starter_code: str | None = None
        eval_spec: EvalSpec = Field(default_factory=EvalSpec)
        native_metadata: NativeMetadata = Field(default_factory=NativeMetadata)
        dedup: DedupMetadata = Field(default_factory=DedupMetadata)
        quality_flags: QualityFlags = Field(default_factory=QualityFlags)

else:

    class _SimpleModel:
        required_fields: ClassVar[set[str]] = set()

        def __init__(self, **data: Any) -> None:
            self.data = data
            for key, value in data.items():
                setattr(self, key, value)

        @classmethod
        def model_validate(cls, data: dict[str, Any]) -> "_SimpleModel":
            if not isinstance(data, dict):
                raise ValueError(f"{cls.__name__} expects a dict")
            missing = [key for key in cls.required_fields if key not in data]
            if missing:
                raise ValueError(f"missing required fields: {missing}")
            return cls(**data)

        def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return deepcopy(self.data)


    class ProblemStatement(_SimpleModel):
        required_fields = {"raw"}


    class Example(_SimpleModel):
        required_fields = {"input", "output"}


    class IOValue(_SimpleModel):
        required_fields = {"kind", "value"}


    class TestCase(_SimpleModel):
        required_fields = {"case_id", "visibility", "input", "expected_output"}


    class ReferenceSolution(_SimpleModel):
        required_fields = {"solution_id", "code"}


    class ComparisonSpec(_SimpleModel):
        pass


    class EvalSpec(_SimpleModel):
        pass


    class NativeMetadata(_SimpleModel):
        pass


    class DedupMetadata(_SimpleModel):
        pass


    class QualityFlags(_SimpleModel):
        pass


    class NormalizedProblem(_SimpleModel):
        required_fields = {
            "problem_id",
            "source",
            "source_problem_id",
            "split",
            "task_family",
            "problem_statement",
            "test_cases",
            "eval_spec",
            "native_metadata",
            "dedup",
            "quality_flags",
        }

        @classmethod
        def model_validate(cls, data: dict[str, Any]) -> "NormalizedProblem":
            errors = validate_problem_dict(data)
            if errors:
                raise ValueError("; ".join(errors))
            return cls(**apply_problem_defaults(data))


def default_comparison() -> dict[str, Any]:
    return {
        "type": "exact_or_token_match",
        "numeric_tolerance": None,
        "case_sensitive": True,
        "strip_trailing_whitespace": True,
    }


def default_eval_spec(eval_mode: str = "stdin_stdout") -> dict[str, Any]:
    return {
        "eval_mode": eval_mode,
        "language": "python",
        "entry_point": None,
        "comparison": default_comparison(),
        "timeout_seconds": 5,
        "memory_limit_mb": 512,
        "requires_special_judge": eval_mode == "special_judge",
        "unsupported_reason": None,
    }


def default_native_metadata() -> dict[str, Any]:
    return {"difficulty": None, "tags": [], "original_task_type": None, "raw_fields": {}}


def default_dedup() -> dict[str, Any]:
    return {
        "normalized_statement_hash": None,
        "canonical_problem_id": None,
        "dedup_group_id": None,
        "is_duplicate": False,
        "duplicate_of": None,
        "dedup_method": None,
    }


def default_quality_flags() -> dict[str, Any]:
    return {
        "has_statement": True,
        "has_tests": True,
        "has_reference_solution": False,
        "has_native_tags": False,
        "is_algorithmic": True,
        "is_supported_for_execution": True,
        "requires_manual_review": False,
        "warnings": [],
    }


def apply_problem_defaults(record: dict[str, Any]) -> dict[str, Any]:
    item = deepcopy(record)
    item.setdefault("schema_version", "0.1")
    item.setdefault("source_url", None)
    item.setdefault("source_platform", None)
    item.setdefault("title", None)
    item.setdefault("examples", [])
    item.setdefault("test_cases", [])
    item.setdefault("reference_solutions", [])
    item.setdefault("starter_code", None)
    item.setdefault("eval_spec", default_eval_spec())
    item["eval_spec"] = {**default_eval_spec(item["eval_spec"].get("eval_mode", "stdin_stdout")), **item["eval_spec"]}
    item["eval_spec"]["comparison"] = {
        **default_comparison(),
        **(item["eval_spec"].get("comparison") or {}),
    }
    item.setdefault("native_metadata", default_native_metadata())
    item["native_metadata"] = {**default_native_metadata(), **item["native_metadata"]}
    item.setdefault("dedup", default_dedup())
    item["dedup"] = {**default_dedup(), **item["dedup"]}
    item.setdefault("quality_flags", default_quality_flags())
    item["quality_flags"] = {**default_quality_flags(), **item["quality_flags"]}
    return item


def validate_problem_dict(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(record, dict):
        return ["record is not an object"]
    for field in NormalizedProblem.required_fields if not PYDANTIC_AVAILABLE else []:
        if field not in record:
            errors.append(f"missing required field: {field}")
    source = record.get("source")
    if source not in SOURCE_VALUES:
        errors.append(f"unsupported source: {source}")
    task_family = record.get("task_family")
    if task_family not in TASK_FAMILY_VALUES:
        errors.append(f"unsupported task_family: {task_family}")
    statement = record.get("problem_statement")
    if not isinstance(statement, dict):
        errors.append("problem_statement must be an object")
    elif "raw" not in statement:
        errors.append("problem_statement.raw is required")
    eval_spec = record.get("eval_spec")
    if not isinstance(eval_spec, dict):
        errors.append("eval_spec must be an object")
    elif eval_spec.get("eval_mode") not in EVAL_MODE_VALUES:
        errors.append(f"unsupported eval_mode: {eval_spec.get('eval_mode')}")
    for idx, case in enumerate(record.get("test_cases") or []):
        if not isinstance(case, dict):
            errors.append(f"test_cases[{idx}] must be an object")
            continue
        if case.get("visibility") not in VISIBILITY_VALUES:
            errors.append(f"test_cases[{idx}].visibility is invalid")
        input_value = case.get("input")
        output_value = case.get("expected_output")
        if not isinstance(input_value, dict) or input_value.get("kind") not in INPUT_KIND_VALUES:
            errors.append(f"test_cases[{idx}].input.kind is invalid")
        if not isinstance(output_value, dict) or output_value.get("kind") not in OUTPUT_KIND_VALUES:
            errors.append(f"test_cases[{idx}].expected_output.kind is invalid")
    return errors


def dump_problem(record: dict[str, Any]) -> dict[str, Any]:
    if PYDANTIC_AVAILABLE:
        return NormalizedProblem.model_validate(record).model_dump(mode="json")
    return NormalizedProblem.model_validate(record).model_dump()
