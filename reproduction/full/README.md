# Full Experimental Reproduction

This directory contains the SLURM templates for regenerating the raw experimental outputs used in the QMHS study.

The full reproduction is intended for an HPC environment. It is **not required** for the lightweight functional check or for regenerating the paper figures and tables from the frozen outputs.

For those workflows, use:

```bash
# Local end-to-end functional check
python reproduction/quick/reproduce_quick.py

# Recreate paper figures/tables from frozen experimental outputs
bash reproduction/reproduce_paper.sh
```

The full workflow instead reruns the computational experiments underlying RQ1, RQ2, and RQ3.

---

## Files

```text
reproduction/full/
├── README.md
├── rq1_minimal.slurm
├── rq2_minimum.slurm
└── rq3_embedding.slurm
```

`rq3_embedding.slurm` is shared by the Zephyr and Pegasus runs. The topology is selected through the `EMBEDDING_TOPOLOGY` environment variable.

---

## Requirements

The full RQ1/RQ2 reproduction requires:

- a SLURM-managed HPC cluster;
- Python with the packages in `requirements.txt` and `requirements-hpc.txt`;
- a working MPI implementation;
- `mpi4py` linked against that MPI implementation;
- the benchmark data, including Git LFS objects.

RQ3 requires the Python dependencies used by the embedding simulator, but does not require MPI or access to a physical D-Wave system.

Before submitting jobs:

```bash
git lfs install
git lfs pull
```

Install the Python dependencies in the environment that will be available on the compute nodes:

```bash
pip install -r requirements.txt
pip install -r requirements-hpc.txt
```

On many HPC systems the MPI implementation itself is provided through the module system, for example:

```bash
module load <cluster-specific-mpi-module>
```

The exact module name is cluster-dependent and is therefore not hard-coded in the supplied SLURM files.

A useful pre-flight check is:

```bash
python -c "from mpi4py import MPI; print(MPI.Get_library_version())"
```

If this fails, the full distributed RQ1/RQ2 jobs will not run.

---

## Repository root and Python environment

Submit the jobs from the repository root whenever possible:

```bash
cd /path/to/QMHS
```

The SLURM files use `SLURM_SUBMIT_DIR` as the repository root by default and verify that both `src/` and `experiments/` exist there.

If the jobs must be submitted from another directory, supply:

```bash
QMHS_REPO_ROOT=/path/to/QMHS
```

for example:

```bash
sbatch --export=ALL,QMHS_REPO_ROOT=/path/to/QMHS \
    reproduction/full/rq1_minimal.slurm
```

The templates use `python3` by default. To use a specific virtual-environment interpreter:

```bash
sbatch --export=ALL,PYTHON=/path/to/venv/bin/python \
    reproduction/full/rq1_minimal.slurm
```

---

## Cluster-specific settings

The supplied SLURM templates preserve the resource profile of the original experiments, but intentionally do not hard-code site-specific values such as:

- partition;
- account/project allocation;
- email address;
- module names;
- local filesystem paths.

If required by the target cluster, add or uncomment directives such as:

```bash
#SBATCH --partition=<partition>
#SBATCH --account=<account>
```

The RQ1 and RQ2 templates use `srun` as the default MPI launcher.

If a cluster requires `mpirun`, submit with:

```bash
sbatch --export=ALL,MPI_LAUNCHER=mpirun \
    reproduction/full/rq1_minimal.slurm
```

The launcher can also include cluster-specific options, provided they are valid for the local environment.

---

# RQ1 — Minimal Hitting Set scalability

Submit:

```bash
sbatch reproduction/full/rq1_minimal.slurm
```

The template requests:

```text
Nodes:             8
MPI tasks:         8
Tasks per node:    1
CPUs per task:     16
Memory:            40 GB
Wall time:         4 days 23 hours
```

The experiment executes:

```text
experiments/rq1_minimal_scalability/scalability_test.py
```

The paper configuration uses:

```text
num_reads  = 2000
num_sweeps = 10000
seeds      = 42, 1337, 690
```

and evaluates universe sizes 5, 10, 25, and 50.

The algorithm variants are:

```text
CardinalitySweep
RecursivePUBO
RecursivePUBO With Preprocessing
RecursivePUBO With Adaptive Qubits
RecursivePUBO With Preprocessing and Adaptive Qubits
```

The experiment writes its benchmark CSV outputs under the output directory defined by the experiment driver.

---

# RQ2 — Minimum Hitting Set scalability

Submit:

```bash
sbatch reproduction/full/rq2_minimum.slurm
```

The template requests:

```text
Nodes:             8
MPI tasks:         8
Tasks per node:    1
CPUs per task:     16
Memory:            15 GB
Wall time:         10 hours
```

The experiment executes:

```text
experiments/rq2_minimum_scalability/scalability_test.py
```

The paper configuration again uses:

```text
num_reads  = 2000
num_sweeps = 10000
seeds      = 42, 1337, 690
```

The evaluated methods are:

```text
BinSearch
LinearSearch
LinearSearch With Preprocessing
LinearSearch With Reduced Qubits
LinearSearch With Preprocessing and Reduced Qubits
```

The experiment writes its benchmark CSV outputs under the output directory defined by the experiment driver.

---

# Classical Hitman baseline

RQ3 compares embedding time against exact Minimum Hitting Set enumeration with the repository's `minimum_baseline_runtime.py` experiment.

Run this in the same software environment used for the other experiments:

```bash
python experiments/classical_baseline/minimum_baseline_runtime.py
```

It evaluates universe sizes 5, 10, 25, and 50 and writes:

```text
hitman_minimum_runtime.csv
```

This computation is independent of the MPI-distributed simulated-annealing jobs.

---

# RQ3 — Embedding feasibility

RQ3 is run once for Zephyr and once for Pegasus.

The template requests:

```text
Nodes:             1
Tasks:             1
CPUs per task:     10
Memory:            64 GB
Wall time:         1 day 23 hours
```

The experiment executes:

```text
experiments/rq3_embedding_feasibility/minimum_embedding/minimum_embedding_test.py
```

The default parameters reproduce the paper embedding configuration:

```text
topology size:             16
embedding attempts/QUBO:   10
timeout/attempt:            30 s
base random seed:          42
local embedding workers:   10
```

## Zephyr

Zephyr is the default:

```bash
sbatch reproduction/full/rq3_embedding.slurm
```

Equivalent explicit submission:

```bash
sbatch --export=ALL,EMBEDDING_TOPOLOGY=zephyr \
    reproduction/full/rq3_embedding.slurm
```

## Pegasus

Run the same experiment with:

```bash
sbatch --export=ALL,EMBEDDING_TOPOLOGY=pegasus \
    reproduction/full/rq3_embedding.slurm
```

The experiment reads the following environment variables:

```text
EMBEDDING_TOPOLOGY
EMBEDDING_TOPOLOGY_SIZE
EMBEDDING_NUM_TRIES
EMBEDDING_TIMEOUT
EMBEDDING_SEED
```

They may be overridden at submission time if needed.

Each topology run exports:

```text
embedding_stats.csv
embedding_attempts.csv
```

The output directory is defined by the RQ3 experiment driver. Preserve the Zephyr and Pegasus outputs separately when regenerating the complete paper dataset.

---

## Recommended execution order

The experiments do not need to be launched through a single master script. A practical full reproduction is:

```text
1. RQ1 distributed scalability experiment
2. RQ2 distributed scalability experiment
3. Hitman minimum-enumeration baseline
4. RQ3 Zephyr embedding experiment
5. RQ3 Pegasus embedding experiment
6. Organize the regenerated raw outputs in the paper-results layout
7. Run reproduction/reproduce_paper.sh
```

RQ1, RQ2, the Hitman baseline, and RQ3 do not depend on one another computationally and may be scheduled independently if cluster resources allow.

---

## Expected paper-results layout

The supplied publication data use the following layout:

```text
data/paper_results/
├── rq1/
│   └── global_stats.csv
├── rq2/
│   └── global_stats.csv
├── rq3/
│   ├── zephyr/
│   │   ├── embedding_attempts.csv
│   │   └── embedding_stats.csv
│   ├── pegasus/
│   │   ├── embedding_attempts.csv
│   │   └── embedding_stats.csv
│   └── hitman_minimum_runtime.csv
└── paper_aggregated/
```

The frozen data shipped with the repository should not be overwritten during routine artifact evaluation. Regenerated full-experiment outputs should first be written to a separate location and compared with the publication data.

---

## Recreating the processed paper results

After the raw outputs have been regenerated and arranged in the expected input layout, the paper analysis pipeline is:

```bash
bash reproduction/reproduce_paper.sh
```

This recreates the summary tables, RQ1/RQ2 figures, Zephyr RQ3 figures, and Pegasus RQ3 figures.

The pipeline concludes by running:

```text
reproduction/check_csvs.py
```

which compares regenerated processed CSVs against the frozen values in:

```text
data/paper_results/paper_aggregated/
```

---

## Important interpretation note

The full workflow regenerates stochastic simulated-annealing and heuristic minor-embedding experiments.

Exact wall-clock runtimes can vary across machines and clusters. Minor-embedding outcomes can also depend on software versions and execution environment despite fixed seeds.

For artifact evaluation, the supplied frozen experimental outputs and the paper-results reproduction workflow are therefore the authoritative route for regenerating the reported figures and aggregate tables. The full HPC workflow is provided to make the original experimental procedure executable and inspectable end to end.
