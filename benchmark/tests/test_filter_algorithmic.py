from __future__ import annotations

import unittest

from acc_data_pipeline.loaders.base import build_problem_record, make_problem_id, make_test_cases
from acc_data_pipeline.preprocessing.filter_algorithmic import filter_records


class FilterAlgorithmicTest(unittest.TestCase):
    def test_keeps_algorithmic_and_removes_output_prediction(self) -> None:
        keep_id = make_problem_id("APPS", "train", "keep")
        keep = build_problem_record(
            source="APPS",
            split="train",
            source_problem_id="keep",
            raw_statement="Given two integers, write a program that prints their sum.",
            test_cases=make_test_cases(keep_id, ["1 2\n"], ["3\n"]),
        )
        remove_id = make_problem_id("LiveCodeBench", "test", "remove")
        remove = build_problem_record(
            source="LiveCodeBench",
            split="test",
            source_problem_id="remove",
            raw_statement="Predict the output of this code snippet.",
            task_family="test_output_prediction",
            test_cases=make_test_cases(remove_id, ["print(1)\n"], ["1\n"]),
            eval_mode="unsupported",
        )
        kept, report = filter_records([keep, remove])
        self.assertEqual([record["problem_id"] for record in kept], [keep["problem_id"]])
        self.assertEqual(report["removed_by_reason"]["test_output_prediction_task"], 1)

    def test_self_repair_requires_candidate_solution(self) -> None:
        problem_id = make_problem_id("LiveCodeBench", "test", "repair")
        record = build_problem_record(
            source="LiveCodeBench",
            split="test",
            source_problem_id="repair",
            raw_statement="Repair this full program so that it prints the sum.",
            task_family="self_repair",
            test_cases=make_test_cases(problem_id, ["1 2\n"], ["3\n"]),
            eval_mode="stdin_stdout",
        )
        kept, report = filter_records([record])
        self.assertEqual(kept, [])
        self.assertEqual(report["removed_by_reason"]["self_repair_missing_candidate_solution"], 1)


if __name__ == "__main__":
    unittest.main()
