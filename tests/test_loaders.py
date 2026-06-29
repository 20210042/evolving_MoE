"""Regression tests for dataset loader behavior.

Uses compact fixtures to verify APPS, CodeContests, TACO, and LiveCodeBench records validate against
the common schema and preserve tricky testcase/tag parsing behavior."""

from __future__ import annotations

import base64
import json
import pickle
import zlib
import unittest
from pathlib import Path

from acc_data_pipeline.loaders.apps_loader import APPSLoader
from acc_data_pipeline.loaders.base import build_problem_record
from acc_data_pipeline.loaders.codecontests_loader import CodeContestsLoader, infer_source_url
from acc_data_pipeline.loaders.livecodebench_loader import (
    LiveCodeBenchLoader,
    decode_livecodebench_test_cases,
    limit_test_payload,
)
from acc_data_pipeline.loaders.taco_loader import TACOLoader
from acc_data_pipeline.schemas.validation import validate_problem


FIXTURES = Path(__file__).parent / "fixtures"


class LoaderTest(unittest.TestCase):
    def assert_valid_records(self, records: list[dict]) -> None:
        self.assertTrue(records)
        for record in records:
            ok, errors = validate_problem(record)
            self.assertTrue(ok, errors)

    def test_apps_loader(self) -> None:
        records = APPSLoader(FIXTURES).load()
        self.assert_valid_records(records)
        self.assertEqual(records[0]["source"], "APPS")
        self.assertEqual(records[0]["eval_spec"]["eval_mode"], "stdin_stdout")

    def test_codecontests_loader(self) -> None:
        records = CodeContestsLoader(FIXTURES).load()
        self.assert_valid_records(records)
        self.assertEqual(records[0]["source"], "CodeContests")
        self.assertEqual(len(records[0]["test_cases"]), 2)

    def test_codecontests_infers_codeforces_url_from_title(self) -> None:
        self.assertEqual(
            infer_source_url({"source": "CODEFORCES", "name": "589_b"}),
            "https://codeforces.com/problemset/problem/589/B",
        )

    def test_taco_loader_function_call(self) -> None:
        records = TACOLoader(FIXTURES).load()
        self.assert_valid_records(records)
        self.assertEqual(records[0]["eval_spec"]["eval_mode"], "function_call")
        self.assertEqual(records[0]["eval_spec"]["entry_point"], "solve")

    def test_livecodebench_loader(self) -> None:
        records = LiveCodeBenchLoader(FIXTURES).load()
        self.assert_valid_records(records)
        self.assertEqual(records[0]["native_metadata"]["original_task_type"], "code_generation")

    def test_livecodebench_decodes_compressed_private_tests(self) -> None:
        cases = [{"input": "1\nabc\n", "output": "YES\n", "testtype": "stdin"}]
        encoded = base64.b64encode(zlib.compress(pickle.dumps(json.dumps(cases)))).decode()
        self.assertEqual(decode_livecodebench_test_cases(encoded), cases)

    def test_livecodebench_test_payload_limit(self) -> None:
        cases = [{"input": f"{index}\n", "output": f"{index}\n"} for index in range(12)]
        self.assertEqual(len(limit_test_payload(cases, 8)), 8)

    def test_statement_sections_do_not_become_examples(self) -> None:
        statement = """Solve the task.

Input

The first line contains n (1 <= n <= 10).

Output

Print the answer.

Examples

Input

3

Output

6

Input

4

Output

10

Note

Simple arithmetic.
"""
        record = build_problem_record(
            source="APPS",
            split="train",
            source_problem_id="statement_sections",
            raw_statement=statement,
        )
        statement_parts = record["problem_statement"]
        self.assertEqual(statement_parts["description"], "Solve the task.")
        self.assertIn("first line contains", statement_parts["input_format"])
        self.assertIn("Print the answer", statement_parts["output_format"])
        self.assertEqual(statement_parts["constraints"], "(1 <= n <= 10)")
        self.assertEqual(record["examples"][0]["input"], "3")
        self.assertEqual(record["examples"][0]["output"], "6")
        self.assertEqual(record["examples"][1]["input"], "4")
        self.assertEqual(record["examples"][1]["output"], "10")

    def test_glued_sample_labels_are_extracted(self) -> None:
        statement = """Can it be sorted?

Input
The first line contains t.

Output
Print YES or NO.Sample Input 1:
2
abc
cba

Sample Output 1:
YES
NO

Note
The first case is already sorted.
"""
        record = build_problem_record(
            source="LiveCodeBench",
            split="test",
            source_problem_id="1873_A",
            raw_statement=statement,
        )
        self.assertEqual(record["problem_statement"]["output_format"], "Print YES or NO.")
        self.assertEqual(record["examples"], [{"input": "2\nabc\ncba", "output": "YES\nNO", "explanation": None}])

    def test_taco_markdown_delimiters_are_extracted(self) -> None:
        statement = """Count placements.

-----Input-----

The only line contains n (5 <= n <= 100).

-----Output-----

Output one integer.

-----Examples-----
Input
5

Output
120
"""
        record = build_problem_record(
            source="TACO",
            split="test",
            source_problem_id="taco_delimiters",
            raw_statement=statement,
        )
        self.assertEqual(record["problem_statement"]["input_format"], "The only line contains n (5 <= n <= 100).")
        self.assertEqual(record["problem_statement"]["constraints"], "(5 <= n <= 100)")
        self.assertEqual(record["examples"][0]["input"], "5")
        self.assertEqual(record["examples"][0]["output"], "120")


if __name__ == "__main__":
    unittest.main()
