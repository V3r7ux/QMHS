#!/bin/bash
#SBATCH --job-name=qa-classical-baseline-minimum
#SBATCH --partition=big

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2

#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=/homes/user/Projects/Internship/logs/qa-classical-baseline-minimum-%j.out
#SBATCH --error=/homes/user/Projects/Internship/logs/qa-classical-baseline-minimum-%j.err
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

python3 experiments/classical_baseline/minimum_baseline_runtime.py