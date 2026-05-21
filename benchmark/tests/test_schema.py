from __future__ import annotations

import unittest

from acc_data_pipeline.loaders.base import build_problem_record, make_problem_id, make_test_cases
from acc_data_pipeline.schemas.problem import NormalizedProblem
from acc_data_pipeline.schemas.validation import validate_problem


class SchemaTest(unittest.TestCase):
    def test_normalized_problem_validates(self) -> None:
        problem_id = make_problem_id("APPS", "train", "0001")
        record = build_problem_record(
            source="APPS",
            split="train",
            source_problem_id="0001",
            raw_statement="Add two integers and print their sum.",
            test_cases=make_test_cases(problem_id, ["1 2\n"], ["3\n"]),
        )
        parsed = NormalizedProblem.model_validate(record)
        dumped = parsed.model_dump(mode="json") if hasattr(parsed, "model_dump") else parsed.model_dump()
        self.assertEqual(dumped["problem_id"], "apps__train__0001")
        ok, errors = validate_problem(record)
        self.assertTrue(ok, errors)

    def test_invalid_eval_mode_is_reported(self) -> None:
        problem_id = make_problem_id("APPS", "train", "0001")
        record = build_problem_record(
            source="APPS",
            split="train",
            source_problem_id="0001",
            raw_statement="Add two integers and print their sum.",
            test_cases=make_test_cases(problem_id, ["1 2\n"], ["3\n"]),
        )
        record["eval_spec"]["eval_mode"] = "made_up"
        ok, errors = validate_problem(record)
        self.assertFalse(ok)
        self.assertTrue(any("eval_mode" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
