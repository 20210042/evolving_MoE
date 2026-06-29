"""Public execution-interface exports.

Makes the unified ExecutionInterface, concrete runners, output comparator, and preparation helper
available from one package-level import."""

from .comparator import compare_outputs
from .execution_interface import ExecutionInterface, prepare_execution_records
from .function_call_runner import FunctionCallRunner
from .stdin_stdout_runner import StdinStdoutRunner

__all__ = [
    "ExecutionInterface",
    "FunctionCallRunner",
    "StdinStdoutRunner",
    "compare_outputs",
    "prepare_execution_records",
]
