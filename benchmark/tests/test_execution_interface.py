from __future__ import annotations

import unittest

from acc_data_pipeline.execution.comparator import compare_outputs
from acc_data_pipeline.execution.execution_interface import ExecutionInterface, prepare_execution_records
from acc_data_pipeline.loaders.base import build_problem_record, make_problem_id, make_test_cases


class ExecutionInterfaceTest(unittest.TestCase):
    def test_comparator_token_match(self) -> None:
        self.assertTrue(compare_outputs("1  2\n", "1 2", {"type": "exact_or_token_match"}))
        self.assertTrue(
            compare_outputs(
                "1.0000001",
                "1.0",
                {"type": "numeric_tolerance", "numeric_tolerance": {"abs_tol": 1e-5}},
            )
        )

    def test_stdin_stdout_runner(self) -> None:
        problem_id = make_problem_id("APPS", "train", "sum")
        problem = build_problem_record(
            source="APPS",
            split="train",
            source_problem_id="sum",
            raw_statement="Given two integers, print their sum.",
            test_cases=make_test_cases(problem_id, ["1 2\n"], ["3\n"]),
        )
        result = ExecutionInterface().run(problem, "a,b=map(int,input().split())\nprint(a+b)\n")
        self.assertTrue(result["passed"], result)

    def test_function_call_runner(self) -> None:
        problem_id = make_problem_id("TACO", "train", "sum")
        problem = build_problem_record(
            source="TACO",
            split="train",
            source_problem_id="sum",
            raw_statement="Return the sum of two integers.",
            test_cases=make_test_cases(problem_id, [[1, 2]], [3], eval_mode="function_call"),
            eval_mode="function_call",
            entry_point="solve",
        )
        result = ExecutionInterface().run(problem, "def solve(a, b):\n    return a + b\n")
        self.assertTrue(result["passed"], result)

    def test_prepare_marks_special_judge_unsupported(self) -> None:
        problem_id = make_problem_id("CodeContests", "train", "spj")
        problem = build_problem_record(
            source="CodeContests",
            split="train",
            source_problem_id="spj",
            raw_statement="Output any valid arrangement.",
            test_cases=make_test_cases(problem_id, ["1\n"], ["1\n"], eval_mode="special_judge"),
            eval_mode="special_judge",
        )
        records, report = prepare_execution_records([problem])
        self.assertEqual(records[0]["eval_spec"]["eval_mode"], "unsupported")
        self.assertEqual(report["unsupported_count"], 1)


if __name__ == "__main__":
    unittest.main()
