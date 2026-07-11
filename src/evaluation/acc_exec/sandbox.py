from __future__ import annotations

import os
import resource
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SandboxResult:
    returncode: int | None
    stdout: str
    stderr: str
    runtime_seconds: float
    timed_out: bool = False
    memory_limit_applied: bool = False


def run_python_code(
    code: str,
    stdin: str = "",
    timeout_seconds: float = 5,
    memory_limit_mb: int | None = 512,
    max_output_bytes: int = 200000,
    filename: str = "candidate.py",
) -> SandboxResult:
    with tempfile.TemporaryDirectory(prefix="acc_eval_") as tmp:
        tmp_path = Path(tmp)
        code_path = tmp_path / filename
        code_path.write_text(code, encoding="utf-8")
        return run_python_file(
            code_path,
            stdin=stdin,
            timeout_seconds=timeout_seconds,
            memory_limit_mb=memory_limit_mb,
            max_output_bytes=max_output_bytes,
            cwd=tmp_path,
        )


def run_python_file(
    path: Path,
    stdin: str = "",
    timeout_seconds: float = 5,
    memory_limit_mb: int | None = 512,
    max_output_bytes: int = 200000,
    cwd: Path | None = None,
) -> SandboxResult:
    start = time.monotonic()
    memory_applied = False

    def preexec() -> None:
        nonlocal memory_applied
        if memory_limit_mb:
            limit = int(memory_limit_mb) * 1024 * 1024
            try:
                resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
                memory_applied = True
            except Exception:
                memory_applied = False

    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(path.parent),
        "PYTHONIOENCODING": "utf-8",
    }
    try:
        completed = subprocess.run(
            [sys.executable, str(path)],
            input=stdin.encode("utf-8", errors="replace"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(cwd or path.parent),
            env=env,
            timeout=timeout_seconds,
            preexec_fn=preexec if os.name == "posix" else None,
        )
        runtime = time.monotonic() - start
        return SandboxResult(
            returncode=completed.returncode,
            stdout=completed.stdout[:max_output_bytes].decode("utf-8", errors="replace"),
            stderr=completed.stderr[:max_output_bytes].decode("utf-8", errors="replace"),
            runtime_seconds=runtime,
            timed_out=False,
            memory_limit_applied=memory_applied,
        )
    except subprocess.TimeoutExpired as exc:
        runtime = time.monotonic() - start
        stdout = (exc.stdout or b"")[:max_output_bytes].decode("utf-8", errors="replace")
        stderr = (exc.stderr or b"")[:max_output_bytes].decode("utf-8", errors="replace")
        return SandboxResult(
            returncode=None,
            stdout=stdout,
            stderr=stderr,
            runtime_seconds=runtime,
            timed_out=True,
            memory_limit_applied=memory_applied,
        )
