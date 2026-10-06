#!/usr/bin/env bash
#SBATCH --job-name=qmhs-rq3-embedding
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=1-23:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Optional cluster-specific directives. Uncomment/adapt if required:
# #SBATCH --partition=<partition>
# #SBATCH --account=<account>
# #SBATCH --mail-user=<email>
# #SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

# ---------------------------------------------------------------------------
# Repository / Python configuration
# ---------------------------------------------------------------------------
#
# Recommended submission:
#   cd /path/to/QMHS
#
# Zephyr (paper main text):
#   sbatch reproduction/full/rq3_embedding.slurm
#
# Pegasus (appendix):
#   sbatch --export=ALL,EMBEDDING_TOPOLOGY=pegasus \
#       reproduction/full/rq3_embedding.slurm
#
# If submitting from another directory, also set:
#   QMHS_REPO_ROOT=/path/to/QMHS
#
REPO_ROOT="${QMHS_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"

if [[ ! -d "$REPO_ROOT/src" || ! -d "$REPO_ROOT/experiments" ]]; then
    echo "ERROR: '$REPO_ROOT' does not look like the QMHS repository root." >&2
    echo "Submit from the repository root or set QMHS_REPO_ROOT=/path/to/QMHS." >&2
    exit 1
fi

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

PYTHON="${PYTHON:-python3}"

if ! command -v "$PYTHON" >/dev/null 2>&1 && [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Python interpreter not found: $PYTHON" >&2
    exit 1
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ---------------------------------------------------------------------------
# RQ3 parameters
# ---------------------------------------------------------------------------
#
# These defaults reproduce the parameters used in the paper experiment.
# They can be overridden at submission time with --export.
#
export EMBEDDING_TOPOLOGY="${EMBEDDING_TOPOLOGY:-zephyr}"
export EMBEDDING_TOPOLOGY_SIZE="${EMBEDDING_TOPOLOGY_SIZE:-16}"
export EMBEDDING_NUM_TRIES="${EMBEDDING_NUM_TRIES:-10}"
export EMBEDDING_TIMEOUT="${EMBEDDING_TIMEOUT:-30}"
export EMBEDDING_SEED="${EMBEDDING_SEED:-42}"

# The Python experiment reads SLURM_CPUS_PER_TASK to choose its local workers.
# This script requests 10 CPUs per task, matching the paper experiment.

# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

echo "QMHS full reproduction — RQ3"
echo "Repository:            $REPO_ROOT"
echo "Job ID:                ${SLURM_JOB_ID:-unset}"
echo "Node:                  ${SLURM_JOB_NODELIST:-unset}"
echo "CPUs per task:         ${SLURM_CPUS_PER_TASK:-unset}"
echo "Python:                $($PYTHON -c 'import sys; print(sys.executable)')"
echo "Embedding topology:    $EMBEDDING_TOPOLOGY"
echo "Topology size:         $EMBEDDING_TOPOLOGY_SIZE"
echo "Embedding attempts:    $EMBEDDING_NUM_TRIES"
echo "Embedding timeout [s]: $EMBEDDING_TIMEOUT"
echo "Embedding base seed:   $EMBEDDING_SEED"

# ---------------------------------------------------------------------------
# Run RQ3
# ---------------------------------------------------------------------------

"$PYTHON" \
    experiments/rq3_embedding_feasibility/minimum_embedding/minimum_embedding_test.py
