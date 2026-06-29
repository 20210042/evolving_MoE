"""Dataset-loader registry.

Maps user-facing dataset names to APPS, CodeContests, TACO, and LiveCodeBench loader classes for the
load CLI and tests."""

from .apps_loader import APPSLoader
from .codecontests_loader import CodeContestsLoader
from .livecodebench_loader import LiveCodeBenchLoader
from .taco_loader import TACOLoader

LOADER_BY_NAME = {
    "apps": APPSLoader,
    "codecontests": CodeContestsLoader,
    "taco": TACOLoader,
    "livecodebench": LiveCodeBenchLoader,
}

__all__ = [
    "APPSLoader",
    "CodeContestsLoader",
    "LOADER_BY_NAME",
    "LiveCodeBenchLoader",
    "TACOLoader",
]
