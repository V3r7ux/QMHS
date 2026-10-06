#!/usr/bin/env bash
#SBATCH --job-name=qmhs-rq1-minimal
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=4-23:00:00
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
#   sbatch reproduction/full/rq1_minimal.slurm
#
# By default SLURM_SUBMIT_DIR is used as the repository root. If you submit
# from another directory, provide the repository explicitly:
#
#   sbatch --export=ALL,QMHS_REPO_ROOT=/path/to/QMHS \
#       reproduction/full/rq1_minimal.slurm
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

# Use an explicitly supplied Python interpreter when desired, e.g.
#   sbatch --export=ALL,PYTHON=/path/to/venv/bin/python ...
PYTHON="${PYTHON:-python3}"

if ! command -v "$PYTHON" >/dev/null 2>&1 && [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Python interpreter not found: $PYTHON" >&2
    exit 1
fi

# Avoid nested BLAS/OpenMP parallelism inside each annealing worker.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

echo "QMHS full reproduction — RQ1"
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

# Prefer Slurm's launcher by default. Override if your cluster requires mpirun:
#   sbatch --export=ALL,MPI_LAUNCHER=mpirun ...
MPI_LAUNCHER="${MPI_LAUNCHER:-srun}"

read -r -a MPI_CMD <<< "$MPI_LAUNCHER"

if ! command -v "${MPI_CMD[0]}" >/dev/null 2>&1; then
    echo "ERROR: MPI launcher not found: ${MPI_CMD[0]}" >&2
    exit 1
fi

echo "MPI launcher:     $MPI_LAUNCHER"

# ---------------------------------------------------------------------------
# Run RQ1
# ---------------------------------------------------------------------------

"${MPI_CMD[@]}" \
    "$PYTHON" \
    experiments/rq1_minimal_scalability/scalability_test.py
