# QMHS

Reproduction package for **“On the Feasibility of Quantum Annealing for Hitting Set Enumeration in Fault Diagnosis.”**

This repository contains the implementations, experiment drivers, benchmark data, frozen paper results, plotting scripts, and reproduction workflows used to evaluate annealing-based formulations for **Minimal Hitting Set (MHS)** and **Minimum Hitting Set** problems arising in spectrum-based fault diagnosis.

The repository supports three levels of reproduction:

1. **Quick local reproduction** — a reduced-budget end-to-end execution of the real pipeline on the small benchmark instances.
2. **Paper-results reproduction** — regeneration of the paper tables and figures from the frozen experimental outputs shipped with the artifact.
3. **Full experimental reproduction** — regeneration of the raw RQ1/RQ2/RQ3 experimental data on an HPC system.

No physical quantum annealer is required. RQ1 and RQ2 use simulated annealing, while RQ3 evaluates minor embedding on idealized Zephyr and Pegasus hardware graphs.

---

## Repository structure

```text
QMHS/
├── src/
│   ├── algorithms/
│   ├── benchmark/
│   ├── core/
│   ├── simulator/
│   └── utils/
│
├── experiments/
│   ├── classical_baseline/
│   ├── rq1_minimal_scalability/
│   ├── rq2_minimum_scalability/
│   └── rq3_embedding_feasibility/
│
├── plotting/
│   ├── create_summary_tables.py
│   ├── main_plots_rq1.py
│   ├── main_plots_rq2.py
│   └── main_plots_rq3.py
│
├── data/
│   ├── spectrasMHS2/
│   ├── mhs2/
│   └── paper_results/
│       ├── rq1/
│       ├── rq2/
│       ├── rq3/
│       │   ├── zephyr/
│       │   └── pegasus/
│       └── paper_aggregated/
│           ├── tables/
│           ├── zephyr/
│           └── pegasus/
│
├── reproduction/
│   ├── quick/
│   │   └── reproduce_quick.py
│   ├── full/
│   │   ├── README.md
│   │   ├── rq1_minimal.slurm
│   │   ├── rq2_minimum.slurm
│   │   └── rq3_embedding.slurm
│   ├── reproduce_paper.sh
│   └── check_csvs.py
│
├── tests/
├── requirements.txt
├── requirements-hpc.txt
└── LICENSE
```

Generated outputs under `reproduction/quick_results/`, `reproduction/figures/`, and `reproduction/tables/` are not part of the frozen artifact and can be regenerated at any time.

---

## Experimental questions

The evaluation is organized around three research questions.

### RQ1 — Minimal Hitting Set enumeration

RQ1 evaluates annealing-based enumeration of inclusion-minimal hitting sets using:

- Cardinality Sweep;
- Recursive PUBO;
- Recursive PUBO with lower-bound preprocessing;
- Recursive PUBO with adaptive coverage ancillas;
- Recursive PUBO with all optimizations.

The paper experiments use simulated annealing with a fixed budget of **2000 reads** and **10000 sweeps** per submitted QUBO.

### RQ2 — Minimum Hitting Set optimization and enumeration

RQ2 evaluates:

- LinearSearch;
- LinearSearch with preprocessing;
- LinearSearch with reduced coverage ancillas;
- LinearSearch with all optimizations;
- BinSearch with all optimizations.

The evaluation distinguishes finding at least one true minimum-cardinality hitting set from recovering the complete set of minimum-cardinality solutions.

### RQ3 — Embedding feasibility

RQ3 evaluates the QUBOs generated for the Minimum Hitting Set problem on idealized:

- **Zephyr** graphs, used for the main-paper analysis;
- **Pegasus** graphs, reported as an additional comparison.

For the paper experiment, each submitted QUBO is embedded using **10 seeded attempts**, a **30 s timeout per attempt**, and base seed **42**. RQ3 evaluates embedding only; no physical QPU execution is performed.

---

## Installation

Clone the repository and retrieve the Git LFS objects:

```bash
git clone https://github.com/V3r7ux/QMHS.git
cd QMHS

git lfs install
git lfs pull
```

Create and activate a Python virtual environment, then install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

For the distributed HPC experiments, install the additional HPC requirements:

```bash
pip install -r requirements-hpc.txt
```

The full RQ1/RQ2 reproduction also requires a working MPI implementation on the cluster, because `mpi4py` must be able to load the system MPI library.

---

# Reproduction workflows

## 1. Quick end-to-end reproduction

The quick reproduction is the recommended first check after installation.

It uses the same parsers, formulations, simulators, decoding, validation, benchmark recording, Hitman baseline, and embedding code as the full experiments, but restricts the benchmark to the small universe-size instances and uses a reduced computational budget.

Run:

```bash
python reproduction/quick/reproduce_quick.py
```

The default quick configuration uses:

```text
Universe sizes:                 5 and 10
Simulated annealing reads:      100
Simulated annealing sweeps:     500
Random seed:                    42
Embedding topologies:           Zephyr and Pegasus
Embedding attempts per QUBO:    2
Embedding timeout per attempt:  5 s
```

For an even smaller installation test:

```bash
python reproduction/quick/reproduce_quick.py \
    --max-problems-per-universe 2 \
    --overwrite
```

The quick workflow writes:

```text
reproduction/quick_results/
├── rq1/
│   ├── global_stats.csv
│   └── length_stats.csv
├── rq2/
│   ├── global_stats.csv
│   └── length_stats.csv
├── rq3_zephyr/
│   ├── embedding_stats.csv
│   └── embedding_attempts.csv
├── rq3_pegasus/
│   ├── embedding_stats.csv
│   └── embedding_attempts.csv
├── hitman_minimum_runtime.csv
└── quick_reproduction_summary.txt
```

The quick workflow is a **functional reproduction**. Its numerical results are not expected to match the paper because the annealing and embedding budgets are intentionally smaller.

---

## 2. Reproduce the paper figures and summary tables

The complete raw outputs used for the publication are included under:

```text
data/paper_results/
```

The processed values used by the paper are frozen under:

```text
data/paper_results/paper_aggregated/
```

To regenerate the paper tables, processed CSVs, and figures from the supplied experimental outputs, run:

```bash
bash reproduction/reproduce_paper.sh
```

This produces generated artifacts under:

```text
reproduction/
├── tables/
└── figures/
    ├── rq1/
    ├── rq2/
    ├── rq3_zephyr/
    └── rq3_pegasus/
```

At the end of the workflow, `reproduction/check_csvs.py` compares the regenerated processed CSVs with the frozen paper aggregates. The check covers the three summary tables together with the RQ1, RQ2, Zephyr, and Pegasus aggregated datasets and exits with a non-zero status if a mismatch is detected.

This is the recommended workflow for reproducing the **reported paper results** without rerunning the multi-day HPC experiments.

---

## 3. Full experimental reproduction

The original raw experiments are substantially more expensive than the quick and paper-results workflows. RQ1 and RQ2 were executed in a distributed CPU environment, while RQ3 uses parallel minor-embedding attempts.

Portable SLURM templates are provided under:

```text
reproduction/full/
```

See:

```text
reproduction/full/README.md
```

for cluster configuration, submission commands, resource requirements, expected outputs, and the order of the full reproduction workflow.

Typical submission from the repository root is:

```bash
sbatch reproduction/full/rq1_minimal.slurm
sbatch reproduction/full/rq2_minimum.slurm

# Zephyr
sbatch reproduction/full/rq3_embedding.slurm

# Pegasus
sbatch --export=ALL,EMBEDDING_TOPOLOGY=pegasus \
    reproduction/full/rq3_embedding.slurm
```

The SLURM templates intentionally leave cluster-specific partition/account settings configurable.

---

## Data

### `data/spectrasMHS2/`

Synthetic spectrum-based fault-localization subjects used to construct the hitting-set instances.

### `data/mhs2/`

Reference MHS2 outputs used as the solution oracle for the synthetic benchmark subjects.

Because these data include large files, the directory is tracked with **Git LFS**. After cloning, run:

```bash
git lfs pull
```

### `data/paper_results/`

Frozen experimental outputs used to generate the paper.

The directory contains the RQ1 and RQ2 scalability results, RQ3 embedding results for Zephyr and Pegasus, the Hitman runtime data used in the embedding comparison, and the frozen processed values used by the figures and summary tables.

These files allow the reported results to be reproduced without rerunning the full HPC experiments.

---

## Tests

Run the repository tests with:

```bash
python -m pytest tests -q
```

The tests cover core formulation, preprocessing, decoding, validation, and algorithmic behavior independently of the large benchmark runs.

---

## Reproducibility levels

The artifact intentionally separates three different goals:

| Workflow | Purpose | Typical environment | Expected cost |
| --- | --- | --- | --- |
| `reproduce_quick.py` | End-to-end functional check | Laptop/workstation | Small |
| `reproduce_paper.sh` | Recreate paper plots/tables from frozen outputs | Laptop/workstation | Minutes |
| `reproduction/full/*.slurm` | Regenerate raw experiment outputs | SLURM HPC cluster | Hours to days |

The quick reproduction should not be used to compare numerical values with the paper. The frozen paper-result workflow is the appropriate path for exact result reproduction, while the full workflow is provided for complete experimental regeneration.

---

## License

This project is distributed under the license included in [`LICENSE`](LICENSE).

---

## Paper

**On the Feasibility of Quantum Annealing for Hitting Set Enumeration in Fault Diagnosis**

If you use this artifact, please cite the accompanying paper. 