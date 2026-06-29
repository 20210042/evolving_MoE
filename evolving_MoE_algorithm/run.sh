#!/bin/bash
# """Handoff note: Thin convenience wrapper for the ACC algorithm pipeline.
# It keeps the repository-level entry point simple and delegates every action to `run_algorithm.sh`,
# so use this file only when you want the default algorithm workflow interface."""

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$REPO/run_algorithm.sh" "$@"
