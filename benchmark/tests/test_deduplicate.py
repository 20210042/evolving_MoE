from __future__ import annotations

import unittest

from acc_data_pipeline.loaders.base import build_problem_record, make_problem_id, make_test_cases
from acc_data_pipeline.preprocessing.deduplicate import deduplicate_records


def record(source: str, source_id: str, statement: str) -> dict:
    problem_id = make_problem_id(source, "train", source_id)
    return build_problem_record(
        source=source,
        split="train",
        source_problem_id=source_id,
        raw_statement=statement,
        test_cases=make_test_cases(problem_id, ["1 2\n"], ["3\n"]),
    )


class DeduplicateTest(unittest.TestCase):
    def test_exact_duplicate_prefers_higher_priority_source(self) -> None:
        apps = record("APPS", "x", "Given two integers, print their sum.")
        taco = record("TACO", "x", "Given two integers, print their sum.")
        output, report = deduplicate_records([apps, taco])
        self.assertEqual(len(output), 1)
        self.assertEqual(output[0]["source"], "TACO")
        self.assertEqual(report["num_removed_duplicates"], 1)

    def test_near_duplicate_detects_token_ngram_overlap(self) -> None:
        left = record(
            "APPS",
            "a",
            "Given two integers a and b, compute and print the sum of a and b.",
        )
        right = record(
            "CodeContests",
            "b",
            "Given two integers a and b, compute and print the sum of a and b please.",
        )
        output, report = deduplicate_records(
            [left, right],
            {"near_duplicate": {"enabled": True, "ngram_size": 3, "threshold": 0.6}},
        )
        self.assertEqual(len(output), 1)
        self.assertEqual(report["num_removed_duplicates"], 1)

    def test_matching_examples_alone_do_not_force_duplicate(self) -> None:
        left = build_problem_record(
            source="APPS",
            split="train",
            source_problem_id="left",
            raw_statement="Count red balls in the box.\n\nExamples\nInput\n1\nOutput\n1",
            examples=[{"input": "1", "output": "1", "explanation": None}],
        )
        right = build_problem_record(
            source="TACO",
            split="train",
            source_problem_id="right",
            raw_statement="Compute the minimum number of operations.\n\nExamples\nInput\n1\nOutput\n1",
            examples=[{"input": "1", "output": "1", "explanation": None}],
        )
        output, report = deduplicate_records([left, right], {"near_duplicate": {"enabled": False}})
        self.assertEqual(len(output), 2)
        self.assertEqual(report["num_removed_duplicates"], 0)


if __name__ == "__main__":
    unittest.main()
