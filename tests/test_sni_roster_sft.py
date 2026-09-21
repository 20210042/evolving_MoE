import json

from train_sft import build_sni_roster_datasets, build_sni_roster_validation_dataset


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_sni_roster_dataset_filters_and_preserves_chat_fields(tmp_path):
    rows = [
        {
            "id": f"row-{index}",
            "system": f"definition-{index}",
            "user": f"examples-and-instruction-{index}",
            "target": f"answer-{index}",
        }
        for index in range(14)
    ]
    train_path = tmp_path / "expert.jsonl"
    split_path = tmp_path / "split.jsonl"
    _write_jsonl(train_path, rows)
    _write_jsonl(
        split_path,
        [
            {"id": row["id"], "n_solved": 0 if index == 0 else index}
            for index, row in enumerate(rows)
        ],
    )

    train, valid, stats = build_sni_roster_datasets(
        str(train_path),
        seed=42,
        n_solved_path=str(split_path),
        max_solved=10,
        eval_ratio=0.2,
        eval_max_samples=2,
    )

    assert stats == {"source": 14, "selected": 11, "filtered": 3, "train": 9, "eval": 2}
    combined = list(train) + list(valid)
    assert {row["id"] for row in combined} == {f"row-{index}" for index in range(11)}
    assert any(row["n_solved"] == 0 for row in combined)
    assert all(row["n_solved"] <= 10 for row in combined)
    row = next(row for row in combined if row["id"] == "row-0")
    assert row["prompt"] == [
        {"role": "system", "content": "definition-0"},
        {"role": "user", "content": "examples-and-instruction-0"},
    ]
    assert row["completion"] == [{"role": "assistant", "content": "answer-0"}]
    assert {row["id"] for row in train}.isdisjoint({row["id"] for row in valid})


def test_sni_roster_split_is_deterministic(tmp_path):
    rows = [
        {"id": str(index), "system": "s", "user": "u", "target": "t"}
        for index in range(20)
    ]
    path = tmp_path / "shared.jsonl"
    _write_jsonl(path, rows)

    first = build_sni_roster_datasets(str(path), seed=7, eval_ratio=0.25, eval_max_samples=512)
    second = build_sni_roster_datasets(str(path), seed=7, eval_ratio=0.25, eval_max_samples=512)

    assert first[0]["id"] == second[0]["id"]
    assert first[1]["id"] == second[1]["id"]


def test_sni_roster_external_validation_matches_export_prompt(tmp_path):
    row = {
        "id": "valid-1",
        "definition": "Choose the label.",
        "answer_line": "Answer with exactly one of: yes, no.",
        "instruction": "is this valid",
        "positive_examples": [
            {"input": "first", "output": "yes"},
            {"input": "second!", "output": "no."},
            {"input": "ignored", "output": "yes"},
        ],
        "ground_truth": ["yes", "acceptable"],
    }
    _write_jsonl(tmp_path / "sni_valid.jsonl", [row])

    dataset = build_sni_roster_validation_dataset(
        str(tmp_path), "valid", seed=42, max_samples=512
    )

    assert len(dataset) == 1
    assert dataset[0]["prompt"][0]["content"].endswith("\n\nChoose the label.")
    assert dataset[0]["prompt"][1]["content"] == (
        "Answer with exactly one of: yes, no.\n\n"
        " Positive Example 1 -\nInput: first.\n Output: yes.\n\n"
        " Positive Example 2 -\nInput: second!\n Output: no.\n\n"
        "Now complete the following example -\nInput: is this valid.\nOutput: "
    )
    assert dataset[0]["completion"] == [{"role": "assistant", "content": "yes"}]
