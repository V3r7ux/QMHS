from __future__ import annotations

"""
Quick end-to-end reproduction for QMHS.

This script exercises the same scientific code paths used by the paper
experiments, but on the |U| = 5 and |U| = 10 benchmark instances and with a
reduced computational budget suitable for a normal workstation.

It runs:

RQ1
    - CardinalitySweep
    - RecursivePUBO
    - RecursivePUBO With Preprocessing
    - RecursivePUBO With Adaptive Qubits
    - RecursivePUBO With Preprocessing and Adaptive Qubits

RQ2
    - BinSearch
    - LinearSearch
    - LinearSearch With Preprocessing
    - LinearSearch With Reduced Qubits
    - LinearSearch With Preprocessing and Reduced Qubits

RQ3
    - the same five Minimum-Hitting-Set formulations
    - ideal Zephyr and Pegasus embedding
    - reduced embedding attempts / timeout

Classical baseline
    - Hitman minimum-hitting-set enumeration runtime

The quick reproduction is a functional end-to-end test. Because it uses a
smaller annealing and embedding budget than the paper experiments, its
numerical results are NOT expected to match the paper results.
"""

import argparse
import csv
import shutil
import time
import sys
from pathlib import Path

GREEN = "\033[92m"
RESET = "\033[0m"


def pass_text() -> str:
    if sys.stdout.isatty():
        return f"{GREEN}PASS{RESET}"
    return "PASS"


ROOT_DIR = Path(__file__).resolve().parents[2]
print(ROOT_DIR)

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tqdm import tqdm

from src.utils.parsing import parse_mhs2_problem

from src.algorithms.quantum.cardinality_sweep import (
    MinimalCardinalitySweepAlgorithm,
    MinimalCardinalitySweepConfig,
)
from src.algorithms.quantum.recursive_pubo import (
    RecursivePUBOAlgorithm,
    RecursivePUBOConfig,
)
from src.algorithms.quantum.minimum_cardinality_sweep import (
    MinimumPUBOAlgorithm,
    MinimumPUBOConfig,
    MinimumSearchMode,
)
from src.algorithms.classical.hitman_minimum import HitmanMinimumOPT

from src.simulator.multinode_simulated_annealing import (
    SimulatedAnnealingConfig,
    SimulatedAnnealingSimulator,
)
from src.simulator.embedding_simulator import (
    EmbeddingFeasibilityConfig,
    EmbeddingFeasibilitySimulator,
)

from src.benchmark.benchmark import BenchmarkRecordSet
from src.benchmark.embedding_benchmark import EmbeddingBenchmarkRecordSet


# ---------------------------------------------------------------------------
# Defaults for the quick reproduction.
# ---------------------------------------------------------------------------

DEFAULT_UNIVERSE_SIZES = [5]
DEFAULT_MAX_NUM_TEST_CASES = 50

DEFAULT_NUM_READS = 100
DEFAULT_NUM_SWEEPS = 500
DEFAULT_SEED = 42
DEFAULT_NUM_THREADS = 1

DEFAULT_EMBEDDING_TOPOLOGIES = ("zephyr", "pegasus")
DEFAULT_EMBEDDING_TOPOLOGY_SIZE = 16
DEFAULT_EMBEDDING_NUM_TRIES = 2
DEFAULT_EMBEDDING_TIMEOUT = 5
DEFAULT_EMBEDDING_NUM_WORKERS = 1

DEFAULT_OUTPUT_DIR = Path("reproduction/quick/quick_results")


# ---------------------------------------------------------------------------
# Algorithm construction.
# ---------------------------------------------------------------------------

def build_rq1_algorithms():
    """Construct exactly the RQ1 algorithm variants used by the paper experiment."""

    cardinality_sweep = MinimalCardinalitySweepAlgorithm(
        MinimalCardinalitySweepConfig(
            prune_test_cases=True,
        )
    )

    recursive_pubo = RecursivePUBOAlgorithm(
        RecursivePUBOConfig(
            prune_test_cases=True,
            use_lower_bound=False,
        )
    )

    recursive_pubo_preprocessed = RecursivePUBOAlgorithm(
        RecursivePUBOConfig(
            prune_test_cases=True,
            use_lower_bound=True,
        )
    )

    recursive_pubo_adaptive = RecursivePUBOAlgorithm(
        RecursivePUBOConfig(
            prune_test_cases=True,
            use_lower_bound=False,
            use_cardinality_limited_coverage_ancillas=True,
        )
    )

    recursive_pubo_all_opt = RecursivePUBOAlgorithm(
        RecursivePUBOConfig(
            prune_test_cases=True,
            use_lower_bound=True,
            use_cardinality_limited_coverage_ancillas=True,
        )
    )

    return [
        ("CardinalitySweep", cardinality_sweep),
        ("RecursivePUBO", recursive_pubo),
        ("RecursivePUBO With Preprocessing", recursive_pubo_preprocessed),
        ("RecursivePUBO With Adaptive Qubits", recursive_pubo_adaptive),
        (
            "RecursivePUBO With Preprocessing and Adaptive Qubits",
            recursive_pubo_all_opt,
        ),
    ]


def build_rq2_algorithms():
    """Construct exactly the RQ2 algorithm variants used by the paper experiment."""

    linear_search = MinimumPUBOAlgorithm(
        MinimumPUBOConfig(
            prune_test_cases=True,
            use_lower_bound=False,
            use_cardinality_limited_coverage_ancillas=False,
            search_mode=MinimumSearchMode.LINEAR_SWEEP,
        )
    )

    linear_search_preprocessed = MinimumPUBOAlgorithm(
        MinimumPUBOConfig(
            prune_test_cases=True,
            use_lower_bound=True,
            use_cardinality_limited_coverage_ancillas=False,
            search_mode=MinimumSearchMode.LINEAR_SWEEP,
        )
    )

    linear_search_reduced = MinimumPUBOAlgorithm(
        MinimumPUBOConfig(
            prune_test_cases=True,
            use_lower_bound=False,
            use_cardinality_limited_coverage_ancillas=True,
            search_mode=MinimumSearchMode.LINEAR_SWEEP,
        )
    )

    linear_search_all_opt = MinimumPUBOAlgorithm(
        MinimumPUBOConfig(
            prune_test_cases=True,
            use_lower_bound=True,
            use_cardinality_limited_coverage_ancillas=True,
            search_mode=MinimumSearchMode.LINEAR_SWEEP,
        )
    )

    binsearch = MinimumPUBOAlgorithm(
        MinimumPUBOConfig(
            prune_test_cases=True,
            use_lower_bound=True,
            use_greedy_upper_bound=True,
            use_cardinality_limited_coverage_ancillas=True,
            search_mode=MinimumSearchMode.BINARY_SEARCH,
        )
    )

    return [
        ("BinSearch", binsearch),
        ("LinearSearch", linear_search),
        ("LinearSearch With Preprocessing", linear_search_preprocessed),
        ("LinearSearch With Reduced Qubits", linear_search_reduced),
        (
            "LinearSearch With Preprocessing and Reduced Qubits",
            linear_search_all_opt,
        ),
    ]


# ---------------------------------------------------------------------------
# Problem loading.
# ---------------------------------------------------------------------------

def load_problem_sets(
    universe_sizes: tuple[int, ...],
    max_num_test_cases: int,
    max_problems_per_universe: int | None,
):
    """
    Load the same MHS2-backed benchmark subjects used by the paper experiments.

    By default all subjects returned by parse_mhs2_problem() for |U| = 5 and 10
    are retained. --max-problems-per-universe can be used for an even smaller
    smoke test while debugging.
    """
    problem_sets = {}

    for universe_size in universe_sizes:
        problems = list(
            parse_mhs2_problem(
                universe_size,
                max_num_test_cases,
                True,
            )
        )

        if max_problems_per_universe is not None:
            problems = problems[:max_problems_per_universe]

        if not problems:
            raise RuntimeError(
                f"No benchmark problems found for universe size {universe_size}."
            )

        problem_sets[universe_size] = problems

    return problem_sets


# ---------------------------------------------------------------------------
# Simulated-annealing solver.
# ---------------------------------------------------------------------------

def build_local_annealing_solver(
    num_reads: int,
    num_sweeps: int,
    num_threads: int,
    seed: int,
):
    """
    Construct the same simulated-annealing backend as the HPC experiments,
    explicitly disabling distributed/MPI execution.
    """
    config = SimulatedAnnealingConfig(
        num_reads=num_reads,
        num_sweeps=num_sweeps,
        num_threads=num_threads,
        seed=seed,
        distributed=False,
        reserve_master_core=False,
    )

    return SimulatedAnnealingSimulator(config)


# ---------------------------------------------------------------------------
# RQ1.
# ---------------------------------------------------------------------------

def run_rq1(
    problem_sets,
    solver,
    output_dir: Path,
) -> None:
    print("\n" + "=" * 72)
    print("RQ1 QUICK REPRODUCTION")
    print("=" * 72)

    algorithms = build_rq1_algorithms()
    records = BenchmarkRecordSet()

    for universe_size, problems in problem_sets.items():
        for problem in tqdm(
            problems,
            desc=f"RQ1 | Universe size {universe_size}",
        ):
            for algorithm_name, algorithm in algorithms:
                start = time.perf_counter()
                result = algorithm.run(problem, solver)
                runtime = time.perf_counter() - start

                stats = result.compute_statistics()

                records.add_statistical_result(
                    problem_id=problem.name,
                    algorithm=algorithm_name,
                    universe_size=problem.size,
                    num_test_cases=len(problem.test_cases),
                    num_qubits=result.num_qubits,
                    num_couplers=result.num_couplers,
                    stats=stats,
                    runtime_seconds=runtime,
                )

    records.export_csv(output_dir)

    validate_benchmark_records(
        records=records,
        expected_algorithms={name for name, _ in algorithms},
        label="RQ1",
    )

    print(f"RQ1 results written to: {output_dir}")


# ---------------------------------------------------------------------------
# RQ2.
# ---------------------------------------------------------------------------

def run_rq2(
    problem_sets,
    solver,
    output_dir: Path,
) -> None:
    print("\n" + "=" * 72)
    print("RQ2 QUICK REPRODUCTION")
    print("=" * 72)

    algorithms = build_rq2_algorithms()
    records = BenchmarkRecordSet()

    for universe_size, problems in problem_sets.items():
        for problem in tqdm(
            problems,
            desc=f"RQ2 | Universe size {universe_size}",
        ):
            for algorithm_name, algorithm in algorithms:
                start = time.perf_counter()
                result = algorithm.run(problem, solver)
                runtime = time.perf_counter() - start

                stats = result.compute_statistics()

                # Unlike the original multi-seed RQ2 benchmark, quick mode uses
                # one fixed seed, so no seed suffix is required in problem_id.
                records.add_statistical_result(
                    problem_id=problem.name,
                    algorithm=algorithm_name,
                    universe_size=problem.size,
                    num_test_cases=len(problem.test_cases),
                    num_qubits=result.num_qubits,
                    num_couplers=result.num_couplers,
                    stats=stats,
                    runtime_seconds=runtime,
                )

    records.export_csv(output_dir)

    validate_benchmark_records(
        records=records,
        expected_algorithms={name for name, _ in algorithms},
        label="RQ2",
    )

    print(f"RQ2 results written to: {output_dir}")


# ---------------------------------------------------------------------------
# Hitman classical baseline.
# ---------------------------------------------------------------------------

def run_hitman_baseline(
    problem_sets,
    output_path: Path,
) -> None:
    print("\n" + "=" * 72)
    print("HITMAN QUICK BASELINE")
    print("=" * 72)

    algorithm = HitmanMinimumOPT()
    rows = []

    for universe_size, problems in problem_sets.items():
        for problem in tqdm(
            problems,
            desc=f"Hitman | Universe size {universe_size}",
        ):
            _, runtime = algorithm.run(problem)

            if runtime < 0:
                raise AssertionError(
                    f"Negative Hitman runtime for problem {problem.name}: {runtime}"
                )

            rows.append(
                {
                    "problem_name": problem.name,
                    "universe_size": problem.size,
                    "num_test_cases": len(problem.test_cases),
                    "runtime_seconds": runtime,
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "problem_name",
                "universe_size",
                "num_test_cases",
                "runtime_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    if not rows:
        raise AssertionError("Hitman baseline produced no rows.")

    print(f"Hitman results written to: {output_path}")


# ---------------------------------------------------------------------------
# RQ3 embedding feasibility.
# ---------------------------------------------------------------------------

def run_rq3_topology(
    problem_sets,
    topology: str,
    topology_size: int,
    num_tries: int,
    timeout: int,
    seed: int,
    num_workers: int,
    output_dir: Path,
) -> None:
    print("\n" + "=" * 72)
    print(f"RQ3 QUICK REPRODUCTION — {topology.upper()}")
    print("=" * 72)

    algorithms = build_rq2_algorithms()
    records = EmbeddingBenchmarkRecordSet()

    config = EmbeddingFeasibilityConfig(
        topology=topology,
        topology_size=topology_size,
        num_tries=num_tries,
        timeout=timeout,
        seed=seed,
        store_embedding=False,
        num_workers=num_workers,
    )

    solver = EmbeddingFeasibilitySimulator(config)

    for universe_size, problems in problem_sets.items():
        for problem in tqdm(
            problems,
            desc=f"RQ3 {topology} | Universe size {universe_size}",
        ):
            solver.set_running_problem(problem)

            for algorithm_name, algorithm in algorithms:
                result_set = algorithm.run(problem, solver)

                records.add_result_set_metadata(
                    problem_id=problem.name,
                    algorithm=algorithm_name,
                    universe_size=problem.size,
                    num_test_cases=len(problem.test_cases),
                    result_metadata=result_set.metadata,
                    store_attempts=True,
                )

    records.export_csv(output_dir)

    validate_embedding_records(
        records=records,
        expected_algorithms={name for name, _ in algorithms},
        expected_topology=topology,
        expected_num_tries=num_tries,
    )

    print(f"RQ3 {topology} results written to: {output_dir}")


# ---------------------------------------------------------------------------
# Validation.
# ---------------------------------------------------------------------------

def validate_benchmark_records(
    records: BenchmarkRecordSet,
    expected_algorithms: set[str],
    label: str,
) -> None:
    """Validate structural invariants of the RQ1/RQ2 quick outputs."""

    if not records.global_records:
        raise AssertionError(f"{label}: no global benchmark records were produced.")

    observed_algorithms = {record.algorithm for record in records.global_records}

    missing_algorithms = expected_algorithms - observed_algorithms
    if missing_algorithms:
        raise AssertionError(
            f"{label}: missing algorithm results: {sorted(missing_algorithms)}"
        )

    for record in records.global_records:
        if record.universe_size < 0:
            raise AssertionError(
                f"{label}: unexpected universe size {record.universe_size}"
            )

        if record.num_test_cases < 0:
            raise AssertionError(
                f"{label}: negative test-case count for {record.problem_id}"
            )

        if record.num_qubits < 0:
            raise AssertionError(
                f"{label}: negative QUBO variable count for {record.problem_id}"
            )

        if record.num_couplers < 0:
            raise AssertionError(
                f"{label}: negative QUBO coupler count for {record.problem_id}"
            )

        if record.total_true_mhs < 0 or record.total_found_mhs < 0:
            raise AssertionError(
                f"{label}: negative solution count for {record.problem_id}"
            )

        if not 0.0 <= record.global_found_ratio <= 1.0:
            raise AssertionError(
                f"{label}: invalid found ratio {record.global_found_ratio} "
                f"for {record.problem_id}"
            )

        if record.runtime_seconds < 0:
            raise AssertionError(
                f"{label}: negative runtime for {record.problem_id}"
            )

    print(
        f"{label} validation: {pass_text()} "
        f"({len(records.global_records)} global records)"
    )


def validate_embedding_records(
    records: EmbeddingBenchmarkRecordSet,
    expected_algorithms: set[str],
    expected_topology: str,
    expected_num_tries: int,
) -> None:
    """Validate structural invariants of the RQ3 quick outputs."""

    if not records.embedding_records:
        raise AssertionError(
            f"RQ3 {expected_topology}: no embedding records were produced."
        )

    if not records.attempt_records:
        raise AssertionError(
            f"RQ3 {expected_topology}: no embedding-attempt records were produced."
        )

    observed_algorithms = {record.algorithm for record in records.embedding_records}
    missing_algorithms = expected_algorithms - observed_algorithms

    if missing_algorithms:
        raise AssertionError(
            f"RQ3 {expected_topology}: missing algorithm results: "
            f"{sorted(missing_algorithms)}"
        )

    for record in records.embedding_records:
        if record.topology.casefold() != expected_topology.casefold():
            raise AssertionError(
                f"RQ3: expected topology {expected_topology!r}, "
                f"found {record.topology!r}"
            )

        if record.num_logical_variables < 0:
            raise AssertionError(
                f"RQ3 {expected_topology}: negative logical-variable count."
            )

        if record.num_logical_couplers < 0:
            raise AssertionError(
                f"RQ3 {expected_topology}: negative logical-coupler count."
            )

        if record.runtime_seconds < 0:
            raise AssertionError(
                f"RQ3 {expected_topology}: negative embedding runtime."
            )

        if record.num_tries != expected_num_tries:
            raise AssertionError(
                f"RQ3 {expected_topology}: expected {expected_num_tries} tries, "
                f"found {record.num_tries}."
            )

        if not 0 <= record.successful_tries <= record.num_tries:
            raise AssertionError(
                f"RQ3 {expected_topology}: invalid successful_tries value."
            )

        if record.embedding_found:
            if record.num_physical_qubits is None:
                raise AssertionError(
                    f"RQ3 {expected_topology}: successful embedding has no "
                    "physical-qubit count."
                )
            if record.mean_chain_length is None:
                raise AssertionError(
                    f"RQ3 {expected_topology}: successful embedding has no "
                    "mean chain length."
                )
            if record.max_chain_length is None:
                raise AssertionError(
                    f"RQ3 {expected_topology}: successful embedding has no "
                    "maximum chain length."
                )

    # Each stored attempt should have a valid attempt id and non-negative runtime.
    for attempt in records.attempt_records:
        if attempt.topology.casefold() != expected_topology.casefold():
            raise AssertionError(
                f"RQ3 attempts: unexpected topology {attempt.topology!r}"
            )

        if not 0 <= attempt.attempt_id < expected_num_tries:
            raise AssertionError(
                f"RQ3 {expected_topology}: invalid attempt_id "
                f"{attempt.attempt_id}."
            )

        if attempt.runtime_seconds < 0:
            raise AssertionError(
                f"RQ3 {expected_topology}: negative attempt runtime."
            )

    print(
        f"RQ3 {expected_topology} validation: {pass_text()} "
        f"({len(records.embedding_records)} submitted QUBOs, "
        f"{len(records.attempt_records)} attempt records)"
    )


def validate_output_files(output_dir: Path, topologies: tuple[str, ...]) -> None:
    """Ensure every expected CSV file was actually written."""

    expected = [
        output_dir / "rq1" / "global_stats.csv",
        output_dir / "rq1" / "length_stats.csv",
        output_dir / "rq2" / "global_stats.csv",
        output_dir / "rq2" / "length_stats.csv",
        output_dir / "hitman_minimum_runtime.csv",
    ]

    for topology in topologies:
        expected.extend(
            [
                output_dir / f"rq3_{topology}" / "embedding_stats.csv",
                output_dir / f"rq3_{topology}" / "embedding_attempts.csv",
            ]
        )

    missing = [path for path in expected if not path.is_file()]

    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise AssertionError(
            "Quick reproduction finished but some expected outputs are missing:\n"
            f"{formatted}"
        )


# ---------------------------------------------------------------------------
# Summary.
# ---------------------------------------------------------------------------

def write_summary(
    output_dir: Path,
    args: argparse.Namespace,
    problem_sets,
) -> None:
    total_problems = sum(len(problems) for problems in problem_sets.values())

    summary_path = output_dir / "quick_reproduction_summary.txt"

    lines = [
        "QMHS quick reproduction",
        "=======================",
        "",
        "Purpose:",
        "  Functional end-to-end reproduction of the paper pipeline.",
        "  Numerical results are not expected to match the paper because quick",
        "  mode intentionally uses a reduced computational budget.",
        "",
        f"Universe sizes: {list(args.universe_sizes)}",
        f"Loaded benchmark subjects: {total_problems}",
        f"Annealing reads per QUBO: {args.num_reads}",
        f"Annealing sweeps per read: {args.num_sweeps}",
        f"Annealing seed: {args.seed}",
        f"Annealing local threads: {args.num_threads}",
        f"Embedding topologies: {list(args.topologies)}",
        f"Embedding topology size: {args.topology_size}",
        f"Embedding attempts per QUBO: {args.embedding_num_tries}",
        f"Embedding timeout per attempt: {args.embedding_timeout} s",
        f"Embedding base seed: {args.seed}",
        f"Embedding workers: {args.embedding_num_workers}",
        "",
        "Status:",
        f"  RQ1: {pass_text()}",
        f"  RQ2: {pass_text()}",
        f"  Hitman baseline: {pass_text()}",
    ]

    for topology in args.topologies:
        lines.append(f"  RQ3 {topology}: {pass_text()}")

    lines.extend(
        [
            "",
            "All quick-reproduction stages completed successfully.",
            "",
        ]
    )

    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(summary_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a local, reduced-budget, end-to-end QMHS reproduction on the "
            "|U|=5 and |U|=10 benchmark subjects."
        )
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )

    parser.add_argument(
        "--universe-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_UNIVERSE_SIZES),
        help="Universe sizes to include (default: 5 10)",
    )

    parser.add_argument(
        "--max-num-test-cases",
        type=int,
        default=DEFAULT_MAX_NUM_TEST_CASES,
        help="Maximum test-case count passed to parse_mhs2_problem (default: 50)",
    )

    parser.add_argument(
        "--max-problems-per-universe",
        type=int,
        default=None,
        help=(
            "Optional cap for debugging. By default ALL benchmark subjects for "
            "each requested universe size are used."
        ),
    )

    parser.add_argument(
        "--num-reads",
        type=int,
        default=DEFAULT_NUM_READS,
        help=f"Simulated-annealing reads per QUBO (default: {DEFAULT_NUM_READS})",
    )

    parser.add_argument(
        "--num-sweeps",
        type=int,
        default=DEFAULT_NUM_SWEEPS,
        help=f"Simulated-annealing sweeps per read (default: {DEFAULT_NUM_SWEEPS})",
    )

    parser.add_argument(
        "--num-threads",
        type=int,
        default=DEFAULT_NUM_THREADS,
        help=f"Local annealing threads (default: {DEFAULT_NUM_THREADS})",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Base random seed (default: {DEFAULT_SEED})",
    )

    parser.add_argument(
        "--topologies",
        nargs="+",
        choices=["zephyr", "pegasus"],
        default=list(DEFAULT_EMBEDDING_TOPOLOGIES),
        help="Embedding topologies (default: zephyr pegasus)",
    )

    parser.add_argument(
        "--topology-size",
        type=int,
        default=DEFAULT_EMBEDDING_TOPOLOGY_SIZE,
        help=(
            "Ideal topology size parameter "
            f"(default: {DEFAULT_EMBEDDING_TOPOLOGY_SIZE})"
        ),
    )

    parser.add_argument(
        "--embedding-num-tries",
        type=int,
        default=DEFAULT_EMBEDDING_NUM_TRIES,
        help=(
            "Embedding attempts per submitted QUBO "
            f"(default: {DEFAULT_EMBEDDING_NUM_TRIES})"
        ),
    )

    parser.add_argument(
        "--embedding-timeout",
        type=int,
        default=DEFAULT_EMBEDDING_TIMEOUT,
        help=(
            "Timeout in seconds for each embedding attempt "
            f"(default: {DEFAULT_EMBEDDING_TIMEOUT})"
        ),
    )

    parser.add_argument(
        "--embedding-num-workers",
        type=int,
        default=DEFAULT_EMBEDDING_NUM_WORKERS,
        help=(
            "Local embedding workers "
            f"(default: {DEFAULT_EMBEDDING_NUM_WORKERS})"
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing quick-results directory before running.",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_fields = {
        "max_num_test_cases": args.max_num_test_cases,
        "num_reads": args.num_reads,
        "num_sweeps": args.num_sweeps,
        "num_threads": args.num_threads,
        "topology_size": args.topology_size,
        "embedding_num_tries": args.embedding_num_tries,
        "embedding_timeout": args.embedding_timeout,
        "embedding_num_workers": args.embedding_num_workers,
    }

    for name, value in positive_fields.items():
        if value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be > 0.")

    if (
        args.max_problems_per_universe is not None
        and args.max_problems_per_universe <= 0
    ):
        raise ValueError("--max-problems-per-universe must be > 0.")


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}\n"
                "Use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    validate_args(args)

    args.universe_sizes = tuple(args.universe_sizes)
    args.topologies = tuple(args.topologies)

    prepare_output_dir(args.output_dir, args.overwrite)

    print("=" * 72)
    print("QMHS QUICK END-TO-END REPRODUCTION")
    print("=" * 72)
    print(
        "This run uses a reduced computational budget and is intended to "
        "validate the full pipeline, not reproduce the paper's numerical values."
    )

    print("\nLoading benchmark subjects...")
    problem_sets = load_problem_sets(
        universe_sizes=args.universe_sizes,
        max_num_test_cases=args.max_num_test_cases,
        max_problems_per_universe=args.max_problems_per_universe,
    )

    for universe_size, problems in problem_sets.items():
        print(f"  |U|={universe_size}: {len(problems)} subjects")

    # One local, non-distributed simulator is shared by RQ1 and RQ2.
    solver = build_local_annealing_solver(
        num_reads=args.num_reads,
        num_sweeps=args.num_sweeps,
        num_threads=args.num_threads,
        seed=args.seed,
    )

    run_rq1(
        problem_sets=problem_sets,
        solver=solver,
        output_dir=args.output_dir / "rq1",
    )

    run_rq2(
        problem_sets=problem_sets,
        solver=solver,
        output_dir=args.output_dir / "rq2",
    )

    run_hitman_baseline(
        problem_sets=problem_sets,
        output_path=args.output_dir / "hitman_minimum_runtime.csv",
    )

    for topology in args.topologies:
        run_rq3_topology(
            problem_sets=problem_sets,
            topology=topology,
            topology_size=args.topology_size,
            num_tries=args.embedding_num_tries,
            timeout=args.embedding_timeout,
            seed=args.seed,
            num_workers=args.embedding_num_workers,
            output_dir=args.output_dir / f"rq3_{topology}",
        )

    validate_output_files(
        output_dir=args.output_dir,
        topologies=args.topologies,
    )

    write_summary(
        output_dir=args.output_dir,
        args=args,
        problem_sets=problem_sets,
    )


if __name__ == "__main__":
    main()
