from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


def deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path | None, defaults: dict[str, Any]) -> dict[str, Any]:
    if not path:
        return deepcopy(defaults)
    path = Path(path)
    if not path.exists():
        return deepcopy(defaults)
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text) or {}
    except ModuleNotFoundError:
        loaded = parse_simple_yaml(text)
    if not isinstance(loaded, dict):
        loaded = {}
    return deep_update(defaults, loaded)


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return {}
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None", "~"}:
        return None
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    try:
        if "." in value or "e" in value.lower():
            return float(value)
        return int(value)
    except ValueError:
        return value


def parse_simple_yaml(text: str) -> dict[str, Any]:
    """Small YAML subset parser for the bundled config files.

    It supports indentation-based dictionaries, scalar values, and block lists.
    PyYAML is used when available; this fallback keeps run.sh usable in the
    minimal conda environment present on the cluster image.
    """

    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]
    pending_key: tuple[int, dict[str, Any], str] | None = None

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if stripped.startswith("- "):
            item = parse_scalar(stripped[2:])
            if pending_key is not None and pending_key[0] == indent:
                _, pending_parent, key = pending_key
                pending_parent[key] = []
                stack.append((indent - 1, pending_parent[key]))
                parent = pending_parent[key]
                pending_key = None
            if isinstance(parent, list):
                parent.append(item)
            continue

        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not isinstance(parent, dict):
            continue
        if value:
            parent[key] = parse_scalar(value)
            pending_key = None
        else:
            parent[key] = {}
            pending_key = (indent + 2, parent, key)
            stack.append((indent, parent[key]))
    return root
