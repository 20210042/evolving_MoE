"""Compatibility exports for test-case related schema objects.

The actual IOValue and TestCase definitions live in problem.py; this module preserves a clearer
import path for code that only cares about test-case structures."""

from .problem import IOValue, TestCase

__all__ = ["IOValue", "TestCase"]
