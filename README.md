# Quantum Annealing Formulations for Hitting Set Enumeration in Fault Diagnosis

This repository contains the replication package for the paper:

> **On the Feasibility of Quantum Annealing for Hitting Set Enumeration in Fault Diagnosis**

The project studies whether quantum annealing can be used as a feasible backend for hitting-set enumeration in software diagnosis. It implements QUBO and PUBO formulations for both **Minimal Hitting Set** enumeration and **Minimum Hitting Set** enumeration, together with preprocessing methods, simulated annealing experiments, exact classical baselines, and minor-embedding experiments on ideal quantum-annealing hardware graphs.

## Overview

In spectrum-based fault diagnosis, failing executions can be represented as conflict sets. A diagnostic candidate must contain at least one component from each conflict, and therefore corresponds to a hitting set.

This repository studies two related enumeration tasks:

* **Minimal Hitting Set enumeration**: recover all inclusion-minimal diagnoses.
* **Minimum Hitting Set enumeration**: recover hitting sets of minimum cardinality.

The implemented formulations include:

* an Exact Formulation QUBO;
* a Cardinality Sweep QUBO;
* a Recursive PUBO formulation for Minimal Hitting Set enumeration;
* a Minimum Hitting Set formulation with linear search and binary search;
* singleton preprocessing;
* dominated-conflict pruning;
* lower-bound preprocessing;
* greedy upper-bound computation;
* adaptive coverage-variable reduction;
* quadratization with ancilla reuse;
* simulated annealing evaluation;
* exact classical minimum-hitting-set enumeration using PySAT Hitman;
* minor-embedding experiments on ideal Pegasus and Zephyr topologies.

The repository is intended to reproduce the empirical evaluation presented in the paper and to make the proposed formulations available for further experimentation.

## Repository structure

```text
QMHS/
  README.md
  requirements.txt
  requirements-hpc.txt
  .gitignore

  src/
    algorithms/
      classical/
      quantum/
    benchmark/
    core/
    preprocessing/
    simulator/
    utils/

  experiments/
    rq1_minimal_scalability/
    rq2_minimum_scalability/
    rq3_embedding_feasibility/
    classical_baseline/

  plotting/
    main_plots_rq1.py
    main_plots_rq2.py
    main_plots_rq3.py

  tests/
    conftest.py
    test_validation.py
    test_preprocessing.py
    test_qubo_formulations.py
    test_hitman_minimum.py

  data/
    spectrasMHS2/
    mhs2/
```

The `src/` directory contains the reusable implementation. The `experiments/` directory contains the scripts used to reproduce the paper experiments. The `plotting/` directory contains the plotting scripts used to generate the paper figures. The `data/` directory contains benchmark instances and reference solutions. The `tests/` directory contains small unit tests for the core logic.

## Installation

Create and activate a Python virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the main dependencies:

```bash
pip install -r requirements.txt
```

The recommended content of `requirements.txt` is:

```text
numpy
scipy
pandas
matplotlib
tqdm
dimod
dwave-neal
python-sat
networkx
dwave-networkx
minorminer
pytest
```

Some parts of the repository have optional dependencies.

For distributed or HPC execution:

```bash
pip install -r requirements-hpc.txt
```

Recommended `requirements-hpc.txt`:

```text
mpi4py
```

## Implemented formulations

### Exact Formulation

Implemented in:

```text
src/algorithms/quantum/exact_formulation.py
```

The Exact Formulation encodes the hitting-set constraints directly as a QUBO. Its ground-state manifold contains all hitting sets of the instance, including non-minimal supersets.

This formulation is useful as a baseline and as a direct encoding of the hitting-set condition, but it is not selective for minimality.

### Cardinality Sweep

Implemented in:

```text
src/algorithms/quantum/cardinality_sweep.py
```

The Cardinality Sweep formulation adds a fixed-cardinality constraint to the Exact Formulation. For each target cardinality `K`, the corresponding QUBO has ground states representing hitting sets of cardinality `K`.

This formulation is used for both Minimal and Minimum Hitting Set experiments. For Minimal Hitting Set enumeration, sampled candidates are post-processed classically to check minimality.

### Recursive PUBO

Implemented in:

```text
src/algorithms/quantum/recursive_pubo.py
```

The Recursive PUBO formulation targets Minimal Hitting Set enumeration. It adds recursive penalties that suppress supersets of previously found minimal hitting sets. These penalties are naturally higher-order polynomial terms, so the implementation quadratizes them into a QUBO using auxiliary variables.

The implementation includes ancilla reuse to reduce the number of auxiliary variables introduced during quadratization.

### Minimum Hitting Set formulation

Implemented in:

```text
src/algorithms/quantum/minimum_cardinality_sweep.py
```

The Minimum Hitting Set formulation searches for hitting sets of minimum cardinality. It supports:

* linear cardinality search;
* binary search over cardinality sectors;
* lower-bound preprocessing;
* greedy upper-bound computation;
* adaptive coverage-variable reduction.

Since every minimum-cardinality hitting set is inclusion-minimal, the recursive PUBO exclusion mechanism is not needed for the Minimum Hitting Set problem.

### Exact classical minimum baseline

Implemented in:

```text
src/algorithms/classical/hitman_minimum.py
```

This baseline uses PySAT Hitman to enumerate all minimum-cardinality hitting sets exactly. It is used in the embedding experiments to compare the runtime of minor embedding against a task-aligned exact classical enumeration baseline.

## Preprocessing utilities

Implemented in:

```text
src/preprocessing/preprocessing_utils.py
```

The preprocessing module includes:

* duplicate test-case removal;
* dominated test-case removal;
* instance validation;
* singleton-solution pruning;
* max-degree lower bound;
* sum-degree lower bound;
* greedy packing lower bound;
* LP-relaxation lower bound;
* greedy upper-bound computation.

These preprocessing methods are used to reduce the size of the generated QUBOs and to reduce the number of cardinality sectors explored by the algorithms.

## Important implementation note: QUBO constant offsets

The mathematical formulations in the paper are written as non-negative penalty Hamiltonians whose valid configurations have energy zero.

The implementation represents QUBOs as coefficient dictionaries and drops constant offsets. Therefore, mathematical zero-energy ground states correspond to shifted energies in the code. For example, a valid hitting set may have negative QUBO energy in the implementation.

This is intentional. Each algorithm provides a `ground_state_energy()` method that returns the expected shifted ground-state energy for the implemented QUBO.

## Running the tests

Run all tests with:

```bash
python -m pytest tests -q
```

The tests check:

* hitting-set validation;
* minimality validation;
* minimum-cardinality extraction;
* singleton preprocessing;
* dominated-conflict pruning;
* lower-bound preprocessing;
* Exact Formulation behavior on the running example;
* Cardinality Sweep behavior on the running example;
* adaptive coverage-variable reduction;
* Recursive PUBO exclusion penalties;
* PySAT Hitman minimum enumeration.

If `python-sat` is not installed, the Hitman tests are skipped.
```

## Checking that the repository compiles

From the repository root, run:

```bash
python -m compileall -q src experiments plotting tests
```

No output means that all Python files compiled successfully.

To compile and test in one command:

```bash
python -m compileall -q src experiments plotting tests && python -m pytest tests -q
```

## Reproducing the experiments

The empirical evaluation is organized around three research questions.

### RQ1: Minimal Hitting Set enumeration scalability

Run:

```bash
python experiments/rq1_minimal_scalability/scalability_test.py
```

This experiment evaluates Minimal Hitting Set enumeration using the Cardinality Sweep and Recursive PUBO formulations under a simulated annealing backend.

The main reported metrics are:

* global found ratio;
* runtime;
* logical variables;
* logical couplers.

### RQ2: Minimum Hitting Set optimization and enumeration

Run:

```bash
python experiments/rq2_minimum_scalability/scalability_test.py
```

This experiment evaluates Minimum Hitting Set recovery using cardinality-based formulations. It compares linear search and binary search, with and without the proposed optimizations.

The main reported metrics are:

* global found ratio over minimum-cardinality solutions;
* minimum-cardinality success;
* runtime;
* logical variables;
* logical couplers.

### RQ3: Minor-embedding feasibility

Run:

```bash
python experiments/rq3_embedding_feasibility/minimum_embedding/minimum_embedding_test.py
```

This experiment constructs the logical QUBOs for the Minimum Hitting Set formulation and attempts to minor-embed them into ideal quantum-annealing hardware graphs.

The embedding experiments do not execute the models on a physical quantum annealer. They measure the feasibility and cost of representing the generated logical QUBOs on sparse quantum-annealing topologies.

The main reported metrics are:

* embedding success;
* number of embedding calls;
* embedding runtime;
* physical qubits;
* mean chain length;
* maximum chain length;
* physical/logical qubit ratio;
* embedding runtime relative to the Hitman baseline.

## Regenerating the figures

After running the corresponding experiments, generate the figures with:

```bash
python plotting/main_plots_rq1.py
python plotting/main_plots_rq2.py
python plotting/main_plots_rq3.py
```

The plotting scripts expect the experiment outputs to be available in the configured output directories.

## Data

The data is available at: TBA

The main data directories contained in the zip archive are:

```text
data/spectrasMHS2/
data/mhs2/
```

`spectrasMHS2/` contains generated conflict instances.

`mhs2/` contains reference minimal hitting sets generated by MHS2.

The reference solutions are used to validate the solutions recovered by the annealing-based formulations.

To run the experiments extract the contents of the `data.zip` archive.

## Reproducibility notes

The simulated annealing experiments are stochastic. The experiment scripts use fixed seeds where applicable, but runtime and sampling behavior may still vary across machines.

The minor-embedding experiments use heuristic embedding algorithms. Failure to find an embedding within the chosen timeout does not prove that no embedding exists. Embedding success should therefore be interpreted as a practical feasibility metric under the specified embedding budget.

The embedding experiments use ideal Pegasus and Zephyr graphs. Real quantum processors may have disabled qubits or couplers, so physical-device embeddability can differ from ideal-topology embeddability.

The RQ3 embedding simulator uses reference solutions only to drive the algorithmic control flow and produce the sequence of QUBOs that would be submitted. The reported RQ3 metrics are embedding metrics, not quantum-annealing solution-quality metrics.


## Citation

Citation information will be added after publication.

```bibtex
@misc{qmhs2026,
  title  = {On the Feasibility of Quantum Annealing for Hitting Set Enumeration in Fault Diagnosis},
  author = {TBA},
  year   = {2026},
  note   = {Replication package},
  url    = {TBA}
}
```

## License

This project is licensed under the MIT License.

See the `LICENSE` file for details.

## Contact

For questions about the replication package, please contact the paper authors.
