#!/usr/bin/env bash
#SBATCH --job-name=qmhs-rq2-minimum
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=15G
#SBATCH --time=10:00:00
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
#   sbatch reproduction/full/rq2_minimum.slurm
#
# If submitting from another directory:
#   sbatch --export=ALL,QMHS_REPO_ROOT=/path/to/QMHS \
#       reproduction/full/rq2_minimum.slurm
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
# Diagnostics
# ---------------------------------------------------------------------------

echo "QMHS full reproduction — RQ2"
echo "Repository:       $REPO_ROOT"
echo "Job ID:           ${SLURM_JOB_ID:-unset}"
echo "Nodes allocated:  ${SLURM_JOB_NUM_NODES:-unset}"
echo "Node list:        ${SLURM_JOB_NODELIST:-unset}"
echo "SLURM_NTASKS:     ${SLURM_NTASKS:-unset}"
echo "Tasks per node:   ${SLURM_NTASKS_PER_NODE:-unset}"
echo "CPUs per task:    ${SLURM_CPUS_PER_TASK:-unset}"
echo "Python:           $($PYTHON -c 'import sys; print(sys.executable)')"

"$PYTHON" -c \
    "from mpi4py import MPI; print('mpi4py MPI library:', MPI.Get_library_version())"

MPI_LAUNCHER="${MPI_LAUNCHER:-srun}"
read -r -a MPI_CMD <<< "$MPI_LAUNCHER"

if ! command -v "${MPI_CMD[0]}" >/dev/null 2>&1; then
    echo "ERROR: MPI launcher not found: ${MPI_CMD[0]}" >&2
    exit 1
fi

echo "MPI launcher:     $MPI_LAUNCHER"

# ---------------------------------------------------------------------------
# Run RQ2
# ---------------------------------------------------------------------------

"${MPI_CMD[@]}" \
    "$PYTHON" \
    experiments/rq2_minimum_scalability/scalability_test.py
