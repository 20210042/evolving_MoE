from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8", errors="replace")


def read_json(path: str | Path, default: Any = None) -> Any:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle, parse_int=str)
    except FileNotFoundError:
        return default
    except json.JSONDecodeError:
        return default


def write_json(path: str | Path, payload: Any) -> None:
    ensure_parent(path)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL record at {path}:{line_no} is not an object")
            yield value


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> int:
    ensure_parent(path)
    count = 0
    with Path(path).open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    return count


def parse_maybe_json(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list, int, float, bool)):
        return value
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return default
        try:
            return json.loads(stripped, parse_int=str)
        except json.JSONDecodeError:
            return value
    return value


def load_json_records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix == ".jsonl":
        return list(iter_jsonl(path))
    value = read_json(path, default=None)
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        for key in ("data", "records", "problems", "items", "examples"):
            nested = value.get(key)
            if isinstance(nested, list):
                return [item for item in nested if isinstance(item, dict)]
        return [value]
    return []


def compact_value(value: Any, max_string: int = 4000, max_list: int = 20) -> Any:
    if isinstance(value, str):
        return value if len(value) <= max_string else value[:max_string] + "...[truncated]"
    if isinstance(value, bytes):
        return compact_value(value.decode("utf-8", errors="replace"), max_string, max_list)
    if isinstance(value, list):
        compacted = [compact_value(item, max_string, max_list) for item in value[:max_list]]
        if len(value) > max_list:
            compacted.append({"_truncated_count": len(value) - max_list})
        return compacted
    if isinstance(value, tuple):
        return compact_value(list(value), max_string, max_list)
    if isinstance(value, dict):
        return {
            str(key): compact_value(item, max_string, max_list)
            for key, item in value.items()
            if not str(key).startswith("_")
        }
    return value
