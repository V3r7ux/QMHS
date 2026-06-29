from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List, FrozenSet, Any
import time
from concurrent.futures import ProcessPoolExecutor
import os

import dimod

try:
    import neal
except ImportError:
    neal = None

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

from src.simulator.base_simulator import BaseSimulator
from src.core.result import SimulatorResult, SimulatorResultSet


TAG_RUN = 1
TAG_STOP = 2
TAG_RESULT = 3


def _run_sa_worker(args):
    """
    Local worker function.

    Each worker receives a BQM and performs a subset of the total reads.
    This is the same idea as the original single-node simulator, but now
    each MPI rank can also run its own local workers.
    """

    bqm, num_reads, num_sweeps, seed = args

    if seed is not None:
        seed = int(seed) % (2**31 - 1)

    sampler = neal.SimulatedAnnealingSampler()

    simulation_results = sampler.sample(
        bqm,
        num_reads=num_reads,
        num_sweeps=num_sweeps,
        seed=seed,
    )

    return simulation_results


def _split_integer(total: int, parts: int) -> List[int]:
    """
    Split an integer as evenly as possible.

    Example
    -------
    total = 10, parts = 3 -> [4, 3, 3]
    """

    if parts <= 0:
        raise ValueError("parts must be positive")

    base = total // parts
    remainder = total % parts

    return [
        base + (1 if i < remainder else 0)
        for i in range(parts)
    ]


@dataclass
class SimulatedAnnealingConfig:
    """
    Configuration container for the simulated annealing solver.

    Parameters
    ----------
    num_reads:
        Total number of reads across all MPI ranks and all local workers.

        Important:
        If you run with 4 nodes and num_reads=2000, the simulator performs
        2000 total reads, not 2000 reads per node.

    num_threads:
        Number of local CPU workers per MPI rank.

        On the master rank, one core is reserved by default for the benchmark
        process itself, so the master rank uses at most num_threads - 1 local
        workers.

    num_sweeps:
        Number of sweeps for each simulated annealing read.

    seed:
        Optional base random seed.

    distributed:
        If True, use MPI when the script is launched with multiple MPI ranks.
        If mpi4py is not available or the world size is 1, the simulator falls
        back to the normal single-node behavior.

    reserve_master_core:
        If True, MPI rank 0 uses one fewer local worker because rank 0 is also
        responsible for running the benchmark script and collecting results.
    """

    num_reads: int = 100
    num_threads: int = 1
    num_sweeps: int = 1000
    seed: int | None = None
    distributed: bool = True
    reserve_master_core: bool = True


class SimulatedAnnealingSimulator(BaseSimulator):
    """
    Simulator backend based on classical simulated annealing.

    This version is distributed-aware.

    Execution model
    ---------------
    - If launched normally with python script.py, it behaves like the original
      simulator.
    - If launched with MPI, for example with:

          srun python script.py

      or:

          mpirun -np 4 python script.py

      then rank 0 runs the benchmark, while ranks > 0 wait inside worker_loop()
      and only execute annealing calls.

    Required benchmark pattern
    --------------------------
    In the benchmark script, after constructing the simulator, add:

        simulator = SimulatedAnnealingSimulator(config)

        if simulator.is_worker:
            simulator.worker_loop()
            return

        # only rank 0 continues with the benchmark

    At the end of the benchmark, call:

        simulator.shutdown_workers()

    ideally inside a finally block.
    """

    def __init__(self, config: SimulatedAnnealingConfig | None = None):
        self.config = config or SimulatedAnnealingConfig()

        if neal is None:
            raise ImportError(
                "Missing dependency 'neal'. Install it with: pip install dwave-neal"
            )

        self.sampler = neal.SimulatedAnnealingSampler()

        self._mpi_enabled = False
        self.comm = None
        self.rank = 0
        self.world_size = 1

        if self.config.distributed and MPI is not None:
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.world_size = self.comm.Get_size()
            self._mpi_enabled = self.world_size > 1

        self._simulation_call_index = 0

    @property
    def is_master(self) -> bool:
        return self.rank == 0

    @property
    def is_worker(self) -> bool:
        return self._mpi_enabled and self.rank != 0

    def _effective_local_threads(self) -> int:
        """
        Number of local workers used by this MPI rank.

        Rank 0 reserves one core by default because it also runs the benchmark
        script and coordinates the distributed annealing calls.
        """

        available_cpus = os.cpu_count() or 1

        requested_threads = max(1, int(self.config.num_threads))
        local_threads = min(requested_threads, available_cpus)

        if (
            self._mpi_enabled
            and self.rank == 0
            and self.config.reserve_master_core
        ):
            local_threads = max(1, local_threads - 1)

        return local_threads

    def _split_reads_between_local_workers(self, num_reads_for_rank: int) -> List[int]:
        """
        Split this rank's reads between local workers.
        """

        local_threads = self._effective_local_threads()
        local_threads = min(local_threads, max(1, num_reads_for_rank))

        read_splits = _split_integer(num_reads_for_rank, local_threads)

        return [r for r in read_splits if r > 0]

    def get_set_from_sample(self, sample) -> FrozenSet[int]:
        """
        Convert a binary sample into the corresponding immutable solution set.

        All indices whose binary value is equal to 1 are included in the
        returned frozenset.

        This preserves the behavior of your original simulator.
        """

        return frozenset(i for i, val in sample.items() if val == 1)

    def _construct_result_from_sample(self, simulation_results: dimod.SampleSet):
        """
        Convert a dimod SampleSet into the internal SimulatorResult list.
        """

        results: List[SimulatorResult] = []

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

    def _make_base_seed(self) -> int:
        """
        Construct a base seed for this simulate() call.

        If the user provided a seed, we deterministically shift it at each
        simulate() call. Otherwise we generate a time-based seed.
        """

        if self.config.seed is None:
            return int(time.time() * 1e6) % (2**31)

        return int(self.config.seed) + 1_000_003 * self._simulation_call_index

    def _run_local_sampling(
        self,
        bqm: dimod.BinaryQuadraticModel,
        num_reads_for_rank: int,
        base_seed: int,
    ) -> dimod.SampleSet | None:
        """
        Run simulated annealing on this MPI rank.

        The assigned reads are split across local CPU workers using
        ProcessPoolExecutor.
        """

        if num_reads_for_rank <= 0:
            return None

        read_splits = self._split_reads_between_local_workers(num_reads_for_rank)

        if len(read_splits) == 0:
            return None

        worker_args = []

        for worker_id, num_reads_i in enumerate(read_splits):
            worker_seed = (
                base_seed
                + 1009 * self.rank
                + 9176 * worker_id
            )

            worker_args.append(
                (
                    bqm,
                    num_reads_i,
                    self.config.num_sweeps,
                    worker_seed,
                )
            )

        if len(worker_args) == 1:
            return _run_sa_worker(worker_args[0]).aggregate()

        with ProcessPoolExecutor(max_workers=len(worker_args)) as executor:
            partial_samplesets = list(executor.map(_run_sa_worker, worker_args))

        return dimod.concatenate(partial_samplesets).aggregate()

    def worker_loop(self) -> None:
        """
        Worker loop for MPI ranks > 0.

        These ranks do not run the benchmark. They wait for BQMs sent by rank 0,
        sample their assigned reads, and send the SampleSet back to rank 0.
        """

        if not self._mpi_enabled:
            return

        if self.rank == 0:
            raise RuntimeError("worker_loop() should not be called on MPI rank 0.")

        while True:
            status = MPI.Status()

            payload = self.comm.recv(
                source=0,
                tag=MPI.ANY_TAG,
                status=status,
            )

            tag = status.Get_tag()

            if tag == TAG_STOP:
                break

            if tag != TAG_RUN:
                raise RuntimeError(
                    f"Rank {self.rank} received unknown MPI tag: {tag}"
                )

            bqm = payload["bqm"]
            num_reads_for_rank = payload["num_reads_for_rank"]
            base_seed = payload["base_seed"]

            local_sampleset = self._run_local_sampling(
                bqm=bqm,
                num_reads_for_rank=num_reads_for_rank,
                base_seed=base_seed,
            )

            self.comm.send(
                local_sampleset,
                dest=0,
                tag=TAG_RESULT,
            )

    def shutdown_workers(self) -> None:
        """
        Stop all MPI worker ranks.

        This should be called once by rank 0 at the end of the benchmark.
        """

        if not self._mpi_enabled:
            return

        if self.rank != 0:
            return

        for worker_rank in range(1, self.world_size):
            self.comm.send(
                None,
                dest=worker_rank,
                tag=TAG_STOP,
            )

    def simulate(self, qubo: Dict[Tuple[int, int], float]) -> SimulatorResultSet:
        """
        Run simulated annealing on the given QUBO and return the sampled results.

        The QUBO is converted into a BinaryQuadraticModel, then sampled using
        the configured number of reads and sweeps.

        In single-node mode, this behaves like the original simulator.

        In distributed mode, rank 0 splits num_reads across MPI ranks. Each MPI
        rank further splits its assigned reads across local CPU workers.
        """

        if self._mpi_enabled and self.rank != 0:
            raise RuntimeError(
                "simulate() must only be called on MPI rank 0. "
                "Worker ranks should be inside worker_loop()."
            )

        bqm = dimod.BinaryQuadraticModel.from_qubo(qubo)

        start = time.time()

        base_seed = self._make_base_seed()
        self._simulation_call_index += 1

        if not self._mpi_enabled:
            simulation_results = self._run_single_node(bqm=bqm, base_seed=base_seed)

            end = time.time()

            results = self._construct_result_from_sample(simulation_results)

            return SimulatorResultSet(
                results=results,
                raw_result=None,
                metadata={
                    "solver": "simulated_annealing",
                    "distributed": False,
                    "num_reads": self.config.num_reads,
                    "num_sweeps": self.config.num_sweeps,
                    "seed": self.config.seed,
                    "num_threads": self.config.num_threads,
                    "effective_local_threads": self._effective_local_threads(),
                    "simulation_time": end - start,
                },
            )

        reads_per_rank = _split_integer(
            self.config.num_reads,
            self.world_size,
        )

        for worker_rank in range(1, self.world_size):
            payload = {
                "bqm": bqm,
                "num_reads_for_rank": reads_per_rank[worker_rank],
                "base_seed": base_seed,
            }

            self.comm.send(
                payload,
                dest=worker_rank,
                tag=TAG_RUN,
            )

        master_sampleset = self._run_local_sampling(
            bqm=bqm,
            num_reads_for_rank=reads_per_rank[0],
            base_seed=base_seed,
        )

        partial_samplesets = []

        if master_sampleset is not None:
            partial_samplesets.append(master_sampleset)

        for worker_rank in range(1, self.world_size):
            worker_sampleset = self.comm.recv(
                source=worker_rank,
                tag=TAG_RESULT,
            )

            if worker_sampleset is not None:
                partial_samplesets.append(worker_sampleset)

        if len(partial_samplesets) == 0:
            raise RuntimeError(
                "No samples were produced. Check num_reads and MPI configuration."
            )

        simulation_results = dimod.concatenate(partial_samplesets).aggregate()

        end = time.time()

        results = self._construct_result_from_sample(simulation_results)

        return SimulatorResultSet(
            results=results,
            raw_result=None,
            metadata={
                "solver": "simulated_annealing",
                "distributed": True,
                "num_reads": self.config.num_reads,
                "num_sweeps": self.config.num_sweeps,
                "seed": self.config.seed,
                "num_threads": self.config.num_threads,
                "mpi_world_size": self.world_size,
                "rank_0_local_threads": self._effective_local_threads(),
                "reads_per_rank": reads_per_rank,
                "total_requested_cpu_workers": self.world_size * self.config.num_threads,
                "reserve_master_core": self.config.reserve_master_core,
                "simulation_time": end - start,
            },
        )

    def _run_single_node(
        self,
        bqm: dimod.BinaryQuadraticModel,
        base_seed: int,
    ) -> dimod.SampleSet:
        """
        Original single-node behavior.

        This is used when MPI is not active or when the script is launched with
        only one MPI rank.
        """

        num_threads = min(
            max(1, int(self.config.num_threads)),
            max(1, int(self.config.num_reads)),
            os.cpu_count() or 1,
        )

        if num_threads == 1:
            simulation_results = self.sampler.sample(
                bqm,
                num_reads=self.config.num_reads,
                num_sweeps=self.config.num_sweeps,
                seed=self.config.seed,
            )

            return simulation_results.aggregate()

        read_splits = _split_integer(self.config.num_reads, num_threads)
        read_splits = [r for r in read_splits if r > 0]

        worker_args = [
            (
                bqm,
                num_reads_i,
                self.config.num_sweeps,
                base_seed + i,
            )
            for i, num_reads_i in enumerate(read_splits)
        ]

        with ProcessPoolExecutor(max_workers=len(worker_args)) as executor:
            partial_samplesets = list(executor.map(_run_sa_worker, worker_args))

        return dimod.concatenate(partial_samplesets).aggregate()