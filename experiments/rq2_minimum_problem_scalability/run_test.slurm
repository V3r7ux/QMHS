#!/bin/bash
#SBATCH --job-name=qa-scalability-minimum
#SBATCH --partition=batch

#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16

#SBATCH --mem=15G
#SBATCH --time=10:00:00
#SBATCH --output=/homes/user/Projects/Internship/logs/qa-scalability-minimum-%j.out
#SBATCH --error=/homes/user/Projects/Internship/logs/qa-scalability-minimum-%j.err
#SBATCH --mail-user=francesco.vertucci22@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

PROJECT_DIR="/homes/user/Projects/Internship/QuantumAnnealer"

cd "$PROJECT_DIR"

mkdir -p /homes/user/Projects/Internship/logs

source ../.qa_venv/bin/activate

export PYTHONUNBUFFERED=1

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export PYTHONPATH="$PROJECT_DIR"

echo "Job ID: $SLURM_JOB_ID"
echo "Nodes allocated: $SLURM_JOB_NUM_NODES"
echo "Node list: $SLURM_JOB_NODELIST"
echo "SLURM_NTASKS: ${SLURM_NTASKS:-unset}"
echo "Tasks per node: ${SLURM_NTASKS_PER_NODE:-unset}"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Submit host: $(hostname)"
echo "Python: $(which python3)"
echo "mpirun: $(which mpirun)"

python3 -c "from mpi4py import MPI; print('mpi4py MPI library:', MPI.Get_library_version())"

mpirun -np "$SLURM_NTASKS" \
    python3 experiments/rq2_minimum_problem_scalability/scalability_test.py