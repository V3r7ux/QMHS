from src.utils.parsing import parse_mhs2_problem
from src.algorithms.quantum.cardinality_sweep import (
    MinimalCardinalitySweepAlgorithm,
    MinimalCardinalitySweepConfig,
)
from src.algorithms.quantum.recursive_pubo import (
    RecursivePUBOAlgorithm,
    RecursivePUBOConfig,
)
from src.simulator.multinode_simulated_annealing import (
    SimulatedAnnealingConfig,
    SimulatedAnnealingSimulator,
)
from src.benchmark.benchmark import BenchmarkRecordSet

from tqdm import tqdm
from pathlib import Path
import time
import os


NUM_READS = 2000
NUM_SWEEPS = 10000

# Number of local CPU workers per MPI rank.
# On rank 0, the simulator will automatically reserve one core for the benchmark script.
NUM_THREADS = int(os.getenv("SLURM_CPUS_PER_TASK", "10"))

SEEDS = [42, 1337, 690]

OUTPUT_DIR = "output/rq1/"


def main():
    # Create simulator immediately.
    #
    # In MPI mode:
    # - rank 0 continues and runs the benchmark;
    # - ranks > 0 enter worker_loop() and wait for annealing calls.
    #
    # In normal single-node mode:
    # - this behaves like the old simulator.
    config = SimulatedAnnealingConfig(
        num_reads=NUM_READS,
        num_sweeps=NUM_SWEEPS,
        num_threads=NUM_THREADS,
        seed=None,  # changed inside the seed loop below
        distributed=True,
        reserve_master_core=True,
    )

    solver = SimulatedAnnealingSimulator(config)

    if solver.is_worker:
        solver.worker_loop()
        return

    try:
        # Only MPI rank 0 reaches this point.

        # Cardinality Sweep Algorithm
        config_algorithm = MinimalCardinalitySweepConfig(
            prune_test_cases=True,
        )
        cardinality_sweep_algorithm = MinimalCardinalitySweepAlgorithm(
            config_algorithm,
        )

        # Recursive PUBO
        config_algorithm = RecursivePUBOConfig(
            prune_test_cases=True,
            use_lower_bound=False,
        )
        recursive_pubo_algorithm = RecursivePUBOAlgorithm(
            config_algorithm,
        )

        # Recursive PUBO with lower bound preprocessing
        config_algorithm = RecursivePUBOConfig(
            prune_test_cases=True,
            use_lower_bound=True,
        )
        preprocessed_recursive_pubo_algorithm = RecursivePUBOAlgorithm(
            config_algorithm,
        )

        # Adaptive Qubits Recursive PUBO
        config_algorithm = RecursivePUBOConfig(
            prune_test_cases=True,
            use_lower_bound=False,
            use_cardinality_limited_coverage_ancillas=True,
        )
        reduced_qubits_algorithm = RecursivePUBOAlgorithm(
            config_algorithm,
        )

        # Recursive PUBO with preprocessing and adaptive qubits
        config_algorithm = RecursivePUBOConfig(
            prune_test_cases=True,
            use_lower_bound=True,
            use_cardinality_limited_coverage_ancillas=True,
        )
        reduced_qubits_preprocessed_algorithm = RecursivePUBOAlgorithm(
            config_algorithm,
        )

        algorithms = [
            (
                "CardinalitySweep",
                cardinality_sweep_algorithm,
            ),
            (
                "RecursivePUBO",
                recursive_pubo_algorithm,
            ),
            (
                "RecursivePUBO With Preprocessing",
                preprocessed_recursive_pubo_algorithm,
            ),
            (
                "RecursivePUBO With Adaptive Qubits",
                reduced_qubits_algorithm,
            ),
            (
                "RecursivePUBO With Preprocessing and Adaptive Qubits",
                reduced_qubits_preprocessed_algorithm,
            ),
        ]

        for seed in SEEDS:
            # Update the simulator seed without reconstructing the simulator.
            #
            # This is important because the MPI workers are already alive and

            # waiting for annealing calls.
            solver.config.seed = seed

            # Create benchmark record
            benchmark_records = BenchmarkRecordSet()

            for universe_size in [5, 10, 25, 50]:
                problems = parse_mhs2_problem(
                    universe_size,
                    50,
                    True,
                )

                desc = f"Seed: {seed} | Universe Size: {universe_size}"

                for problem in tqdm(problems[:80], desc=desc):
                    for algorithm_name, algorithm in algorithms:
                        start = time.time()

                        r = algorithm.run(problem, solver)

                        end = time.time()

                        sr = r.compute_statistics()

                        benchmark_records.add_statistical_result(
                            problem_id=problem.name,
                            algorithm=algorithm_name,
                            universe_size=problem.size,
                            num_test_cases=len(problem.test_cases),
                            num_qubits=r.num_qubits,
                            num_couplers=r.num_couplers,
                            stats=sr,
                            runtime_seconds=end - start,
                        )

            benchmark_records.export_csv(Path(OUTPUT_DIR) / str(seed))

    finally:
        # Tell all MPI worker ranks to stop.
        #
        # This must be called by rank 0 once the benchmark is finished,
        # otherwise worker ranks remain blocked inside worker_loop().
        solver.shutdown_workers()


if __name__ == "__main__":
    main()