from src.utils.parsing import parse_mhs2_problem

from src.algorithms.quantum.minimum_cardinality_sweep import (
    MinimumPUBOConfig,
    MinimumPUBOAlgorithm,
    MinimumSearchMode,
)

from src.simulator.embedding_simulator import (
    EmbeddingFeasibilityConfig,
    EmbeddingFeasibilitySimulator,
)

from src.benchmark.embedding_benchmark import EmbeddingBenchmarkRecordSet

from tqdm import tqdm

import os
from pathlib import Path


TOPOLOGY = os.getenv("EMBEDDING_TOPOLOGY", "zephyr")
TOPOLOGY_SIZE = int(os.getenv("EMBEDDING_TOPOLOGY_SIZE", "16"))

NUM_TRIES = int(os.getenv("EMBEDDING_NUM_TRIES", "10"))
TIMEOUT = int(os.getenv("EMBEDDING_TIMEOUT", "30"))
SEED = int(os.getenv("EMBEDDING_SEED", "42"))

NUM_THREADS = int(os.getenv("SLURM_CPUS_PER_TASK", "10"))

UNIVERSE_SIZES = [5, 10, 25, 50]
MAX_NUM_TEST_CASES = 50

OUTPUT_DIR = "output/rq3"


def main():
    """
    Run the embedding-feasibility benchmark for the minimum MHS algorithms.

    This benchmark does not perform simulated annealing. Each algorithm
    constructs its QUBO/BQM as usual and passes it to the embedding-feasibility
    simulator, which tries to minor-embed the logical QUBO graph onto an ideal
    hardware topology.

    The benchmark output is written to:

        output/embeddign_feasability/minimum_embedding

    The exported files are:

        embedding_stats.csv
        embedding_attempts.csv
    """

    benchmark_records = EmbeddingBenchmarkRecordSet()

    # Recursive PUBO
    config_algorithm = MinimumPUBOConfig(
        prune_test_cases=True, 
        use_lower_bound=False, 
        use_cardinality_limited_coverage_ancillas=False,
        search_mode=MinimumSearchMode.LINEAR_SWEEP
    )
    linear_search_algorithm = MinimumPUBOAlgorithm(config_algorithm)

    # Preprocessing Recursive Pubo
    config_algorithm = MinimumPUBOConfig(
        prune_test_cases=True, 
        use_lower_bound=True, 
        use_cardinality_limited_coverage_ancillas=False,
        search_mode=MinimumSearchMode.LINEAR_SWEEP
    )
    preprocessed_algorithm = MinimumPUBOAlgorithm(config_algorithm)

    # Reduced Qubits Recursive Pubo
    config_algorithm = MinimumPUBOConfig(
        prune_test_cases=True, 
        use_lower_bound=False, 
        use_cardinality_limited_coverage_ancillas=True,
        search_mode=MinimumSearchMode.LINEAR_SWEEP
    )
    reduced_qubits_algorithm = MinimumPUBOAlgorithm(config_algorithm)

    # Reduced Qubits Preprocessing Recursive Pubo
    config_algorithm = MinimumPUBOConfig(
        prune_test_cases=True, 
        use_lower_bound=True, 
        use_cardinality_limited_coverage_ancillas=True,
        search_mode=MinimumSearchMode.LINEAR_SWEEP
    )
    reduced_qubits_preprocessed_algorithm = MinimumPUBOAlgorithm(config_algorithm)

    # Binary Search Algorithm
    config_algorithm = MinimumPUBOConfig(
        prune_test_cases=True, 
        use_lower_bound=True,
        use_greedy_upper_bound=True, 
        use_cardinality_limited_coverage_ancillas=True,
        search_mode=MinimumSearchMode.BINARY_SEARCH
    )
    binsearch_algorithm = MinimumPUBOAlgorithm(config_algorithm)

    algorithms = [
        (
            "BinSearch",
            binsearch_algorithm,
        ),
        (
            "LinearSearch",
            linear_search_algorithm,
        ),
        (
            "LinearSearch With Preprocessing",
            preprocessed_algorithm,
        ),
        (
            "LinearSearch With Reduced Qubits",
            reduced_qubits_algorithm,
        ),
        (
            "LinearSearch With Preprocessing and Reduced Qubits",
            reduced_qubits_preprocessed_algorithm,
        ),
    ]

    # Create the embedding-feasibility simulator.
    #
    # This replaces the annealing sampler. It returns no solutions, but stores
    # all embedding-related information in the simulator metadata.
    config = EmbeddingFeasibilityConfig(
        topology=TOPOLOGY,
        topology_size=TOPOLOGY_SIZE,
        num_tries=NUM_TRIES,
        timeout=TIMEOUT,
        seed=SEED,
        store_embedding=False,
        num_workers=NUM_THREADS,
    )

    solver = EmbeddingFeasibilitySimulator(config)

    for universe_size in UNIVERSE_SIZES:
        problems = parse_mhs2_problem(
            universe_size,
            MAX_NUM_TEST_CASES,
            True,
        )

        for problem in tqdm(
            problems,
            desc=f"Minimum embedding feasibility | Universe Size: {universe_size}",
        ):

            solver.set_running_problem(problem)

            for algorithm_name, algorithm in algorithms:
                result_set = algorithm.run(problem, solver)

                benchmark_records.add_result_set_metadata(
                    problem_id=problem.name,
                    algorithm=algorithm_name,
                    universe_size=problem.size,
                    num_test_cases=len(problem.test_cases),
                    result_metadata=result_set.metadata,
                    store_attempts=True,
                )

    benchmark_records.export_csv( Path(OUTPUT_DIR) / TOPOLOGY)


if __name__ == "__main__":
    main()