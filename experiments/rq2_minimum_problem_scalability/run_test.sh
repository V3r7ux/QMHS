#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT"

python "$PROJECT_ROOT/experiments/rq2_minimum_problem_scalability/scalability_test.py"
