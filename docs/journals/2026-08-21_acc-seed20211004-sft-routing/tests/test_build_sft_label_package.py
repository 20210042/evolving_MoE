import json
import subprocess
import sys
from pathlib import Path


def test_build_sft_label_package_preserves_personas_and_labels(tmp_path):
    roster = tmp_path / "roster.json"
    binned = tmp_path / "binned.jsonl"
    source = tmp_path / "source.jsonl"
    output = tmp_path / "package"
    roster.write_text(json.dumps([
        {"id": "a", "name": "Alpha", "system_prompt": "Persona A"},
        {"id": "b", "name": "Beta", "system_prompt": "Persona B"},
    ]), encoding="utf-8")
    binned.write_text(json.dumps({
        "id": "p1", "dataset": "acc", "n_solved": 1,
        "per_expert": {"a": 1, "b": 0},
    }) + "\n", encoding="utf-8")
    source.write_text(json.dumps({
        "id": "p1", "instruction": "Solve it", "solution": "print(1)",
    }) + "\n", encoding="utf-8")

    script = Path(__file__).parents[1] / "scripts" / "build_sft_label_package.py"
    subprocess.run([
        sys.executable, str(script),
        "--roster", str(roster),
        "--binned", str(binned),
        "--source-jsonl", str(source),
        "--output-dir", str(output),
    ], check=True)

    mapping = json.loads((output / "agent_mapping.json").read_text(encoding="utf-8"))
    label = json.loads((output / "binning_labels.jsonl").read_text(encoding="utf-8"))
    assert mapping["a"]["system_prompt"] == "Persona A"
    assert mapping["a"]["train_pass_at_1"] == 1.0
    assert mapping["b"]["train_pass_at_1"] == 0.0
    assert label["per_expert"] == {"a": 1, "b": 0}
