"""Execution-result schema for candidate solution evaluation.

Defines the normalized status values, failed-test payload, and result model used by
all runners so downstream reports can compare stdin/stdout and function-call tasks uniformly."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, ClassVar, Literal

try:
    from pydantic import BaseModel, ConfigDict, Field

    PYDANTIC_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover
    BaseModel = object  # type: ignore
    ConfigDict = dict  # type: ignore
    Field = None  # type: ignore
    PYDANTIC_AVAILABLE = False


STATUS_VALUES = {
    "accepted",
    "wrong_answer",
    "runtime_error",
    "timeout",
    "compile_error",
    "memory_limit_exceeded",
    "unsupported_language",
    "unsupported_eval_mode",
    "internal_error",
}


if PYDANTIC_AVAILABLE:

    class ACCBaseModel(BaseModel):
        model_config = ConfigDict(extra="allow")


    class FailedTest(ACCBaseModel):
        case_id: str
        input: dict[str, Any]
        expected_output: dict[str, Any]
        actual_output: dict[str, Any] | None = None
        error_message: str | None = None


    class ExecutionResult(ACCBaseModel):
        problem_id: str
        solution_id: str = "candidate"
        passed: bool
        status: Literal[
            "accepted",
            "wrong_answer",
            "runtime_error",
            "timeout",
            "compile_error",
            "memory_limit_exceeded",
            "unsupported_language",
            "unsupported_eval_mode",
            "internal_error",
        ]
        num_tests_passed: int = 0
        num_tests_total: int = 0
        failed_tests: list[FailedTest] = Field(default_factory=list)
        runtime_seconds: float | None = None
        memory_mb: float | None = None
        stderr: str | None = None

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
            if "status" in data and data["status"] not in STATUS_VALUES:
                raise ValueError(f"unsupported status: {data['status']}")
            return cls(**data)

        def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return deepcopy(self.data)


    class FailedTest(_SimpleModel):
        required_fields = {"case_id", "input", "expected_output"}


    class ExecutionResult(_SimpleModel):
        required_fields = {"problem_id", "passed", "status"}
