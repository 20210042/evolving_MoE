"""Convenience exports for preprocessing stages.

Expose normalization, algorithmic filtering, and deduplication functions as the main building
blocks used by the CLI pipeline."""

from .deduplicate import deduplicate_records
from .filter_algorithmic import filter_records
from .normalize import normalize_record

__all__ = ["deduplicate_records", "filter_records", "normalize_record"]
