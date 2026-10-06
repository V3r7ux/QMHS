from src.utils.parsing import parse_mhs2_problem
from src.algorithms.quantum.minimum_cardinality_sweep import (
    MinimumPUBOConfig,
    MinimumPUBOAlgorithm,
    MinimumSearchMode,
)
from src.simulator.multinode_simulated_annealing import (
    SimulatedAnnealingConfig,
    SimulatedAnnealingSimulator,
)
from src.benchmark.benchmark import BenchmarkRecordSet

from tqdm import tqdm
import time
import os


NUM_READS = 2000
NUM_SWEEPS = 10000

# Number of local CPU workers per MPI rank.
# On rank 0, the simulator reserves one core for the benchmark script.
NUM_THREADS = int(os.getenv("SLURM_CPUS_PER_TASK", "10"))

SEEDS = [42, 1337, 690]

OUTPUT_DIR = "output/rq2/"


def main():
    # Create simulator immediately.
    #
    # In MPI mode:
    # - rank 0 runs the benchmark;
    # - ranks > 0 enter worker_loop() and only execute annealing calls.
    #
    # In single-node mode:
    # - this behaves like the old simulator.
    config = SimulatedAnnealingConfig(
        num_reads=NUM_READS,
        num_sweeps=NUM_SWEEPS,
        num_threads=NUM_THREADS,
        seed=None,  # updated inside the seed loop
        distributed=True,
        reserve_master_core=True,
    )

    solver = SimulatedAnnealingSimulator(config)

    if solver.is_worker:
        solver.worker_loop()
        return

    try:
        # Only rank 0 reaches this point.

        benchmark_records = BenchmarkRecordSet()

        # Linear Search
        config_algorithm = MinimumPUBOConfig(
            prune_test_cases=True,
            use_lower_bound=False,
            use_cardinality_limited_coverage_ancillas=False,
            search_mode=MinimumSearchMode.LINEAR_SWEEP,
        )
        linear_search_algorithm = MinimumPUBOAlgorithm(config_algorithm)

        # Linear Search with preprocessing
        config_algorithm = MinimumPUBOConfig(
            prune_test_cases=True,
            use_lower_bound=True,
            use_cardinality_limited_coverage_ancillas=False,
            search_mode=MinimumSearchMode.LINEAR_SWEEP,
        )
        preprocessed_algorithm = MinimumPUBOAlgorithm(config_algorithm)

        # Linear Search with reduced qubits
        config_algorithm = MinimumPUBOConfig(
            prune_test_cases=True,
            use_lower_bound=False,
            use_cardinality_limited_coverage_ancillas=True,
            search_mode=MinimumSearchMode.LINEAR_SWEEP,
        )
        reduced_qubits_algorithm = MinimumPUBOAlgorithm(config_algorithm)

        # Linear Search with preprocessing and reduced qubits
        config_algorithm = MinimumPUBOConfig(
            prune_test_cases=True,
            use_lower_bound=True,
            use_cardinality_limited_coverage_ancillas=True,
            search_mode=MinimumSearchMode.LINEAR_SWEEP,
        )
        reduced_qubits_preprocessed_algorithm = MinimumPUBOAlgorithm(config_algorithm)

        # Binary Search
        config_algorithm = MinimumPUBOConfig(
            prune_test_cases=True,
            use_lower_bound=True,
            use_greedy_upper_bound=True,
            use_cardinality_limited_coverage_ancillas=True,
            search_mode=MinimumSearchMode.BINARY_SEARCH,
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

        for seed in SEEDS:
            # Update the simulator seed without reconstructing the simulator.
            # The MPI workers are already alive and waiting for annealing jobs.
            solver.config.seed = seed

            for universe_size in [5, 10, 25, 50]:
                problems = parse_mhs2_problem(
                    universe_size,
                    50,
                    True,
                )

                desc = f"Seed: {seed} | Universe Size: {universe_size}"

                for problem in tqdm(problems, desc=desc):
                    for algorithm_name, algorithm in algorithms:
                        start = time.time()

                        r = algorithm.run(problem, solver)

                        end = time.time()

                        sr = r.compute_statistics()

                        benchmark_records.add_statistical_result(
                            problem_id=problem.name + str(seed),
                            algorithm=algorithm_name,
                            universe_size=problem.size,
                            num_test_cases=len(problem.test_cases),
                            num_qubits=r.num_qubits,
                            num_couplers=r.num_couplers,
                            stats=sr,
                            runtime_seconds=end - start,
                        )

        benchmark_records.export_csv(OUTPUT_DIR)

    finally:
        # Stop all MPI worker ranks.
        # Without this, workers remain blocked inside worker_loop().
        solver.shutdown_workers()


if __name__ == "__main__":
    main()