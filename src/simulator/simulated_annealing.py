from dataclasses import dataclass
from typing import Dict, Tuple, List, FrozenSet
import time
from concurrent.futures import ProcessPoolExecutor
import os

import dimod

try:
    import neal
except ImportError:
    neal = None

from src.simulator.base_simulator import BaseSimulator
from src.core.result import SimulatorResult, SimulatorResultSet

def _run_sa_worker(args):

    bqm, num_reads, num_sweeps, seed = args

    sampler = neal.SimulatedAnnealingSampler() 

    simulation_results = sampler.sample(
            bqm,
            num_reads=num_reads,
            num_sweeps=num_sweeps,
            seed=seed,
        )
    
    return simulation_results


@dataclass
class SimulatedAnnealingConfig:
    """
    Configuration container for the simulated annealing solver.

    It defines the number of reads, the number of threds to split the reads, the number of sweeps performed in each run,
    and an optional random seed to make the sampling process reproducible.
    """
    num_reads: int = 100
    num_threads: int = 1
    num_sweeps: int = 1000
    seed: int | None = None


class SimulatedAnnealingSimulator(BaseSimulator):
    """
    Simulator backend based on classical simulated annealing.

    This class uses the neal sampler to solve a QUBO formulation through
    repeated annealing runs. It converts the sampled binary configurations
    into solution sets and returns the aggregated results together with
    execution metadata.
    """

    def __init__(self, config: SimulatedAnnealingConfig | None = None):
        """
        Initialize the simulated annealing simulator with the given configuration.

        If no configuration is provided, a default one is created. The method
        also checks that the neal dependency is available and initializes the
        underlying simulated annealing sampler.
        """
        self.config = config or SimulatedAnnealingConfig()

        if neal is None:
            raise ImportError(
                "Missing dependency 'neal'. Install it with: pip install dwave-neal"
            )

        self.sampler = neal.SimulatedAnnealingSampler()

    def _split_reads_between_workers(self) -> List[int]:
        """
        Split total_reads as evenly as possible across processes.
        Example: 10 reads on 3 processes -> [4, 3, 3]
        """
        base = self.config.num_reads // self.config.num_threads
        remainder = self.config.num_reads % self.config.num_threads
        return [base + (1 if i < remainder else 0) for i in range(self.config.num_threads)]


    def get_set_from_sample(self, sample) -> FrozenSet[int]:
        """
        Convert a binary sample into the corresponding immutable solution set.

        All indices whose binary value is equal to 1 are included in the
        returned frozenset.
        """
        return frozenset(i for i, val in sample.items() if val == 1)
    
    def _construct_result_from_sample(self, simulation_results: dimod.SampleSet):
        # Construct results object
        results: List[SimulatorResult] = []

        # Contruct return object
        for record in simulation_results.data():
            sample = {int(k): int(v) for k, v in record.sample.items()}
            energy = float(record.energy)
            num_occurrences = getattr(record, "num_occurrences", None)

            results.append(
                SimulatorResult(
                    sample=self.get_set_from_sample(sample),
                    energy=energy,
                    num_occurrences=num_occurrences,
                )
            )
            
        return results


    def simulate(self, qubo: Dict[Tuple[int, int], float]) -> SimulatorResultSet:
        """
        Run simulated annealing on the given QUBO and return the sampled results.

        The QUBO is first converted into a binary quadratic model, then sampled
        using the configured number of reads and sweeps. Equivalent solutions
        are aggregated by summing their occurrence counts. The method returns
        the processed results, the raw sampler output, and metadata describing
        the execution.
        """

        # Convert qubo matrix to bqm
        bqm = dimod.BinaryQuadraticModel.from_qubo(qubo)

        start = time.time()

        num_threads = min(self.config.num_threads, self.config.num_reads, os.cpu_count())
        
        if num_threads == 1:

            # Run simulation and sample results
            simulation_results = self.sampler.sample(
                bqm,
                num_reads=self.config.num_reads,
                num_sweeps=self.config.num_sweeps,
                seed=self.config.seed,
            )

        else:
            # Split reads between workers
            read_splits = self._split_reads_between_workers()
            read_splits = [r for r in read_splits if r>0]

            base_seed = self.config.seed if self.config.seed is not None else int(time.time() * 1e6) % (2**31)

            worker_args = [
                (
                    bqm,
                    num_reads_i,
                    self.config.num_sweeps,
                    base_seed+i
                )
                for i, num_reads_i in enumerate(read_splits)
            ]

            with ProcessPoolExecutor(max_workers=len(worker_args)) as executor:
                partial_samplesets = list(executor.map(_run_sa_worker, worker_args))
            
            simulation_results = dimod.concatenate(partial_samplesets).aggregate()

        end = time.time()

        # Construct results object
        results = self._construct_result_from_sample(simulation_results)

        return SimulatorResultSet(
            results=results,
            raw_result=simulation_results,
            metadata={
                "solver": "simulated_annealing",
                "num_reads": self.config.num_reads,
                "num_sweeps": self.config.num_sweeps,
                "seed": self.config.seed,
                "simulation_time": end - start
            },
        )