from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import os
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

import dimod
import dwave_networkx as dnx
import minorminer
import networkx as nx

from src.simulator.base_simulator import BaseSimulator
from src.core.result import SimulatorResultSet, SimulatorResult


def _run_minorminer_attempt_worker(
    source_graph: nx.Graph,
    target_graph: nx.Graph,
    attempt_id: int,
    timeout: int | None,
    seed: int | None,
) -> dict[str, Any]:
    """
    Run one minorminer attempt in a separate process.

    This function is defined at module level because multiprocessing needs
    worker functions to be picklable.
    """

    kwargs: dict[str, Any] = {}

    if timeout is not None:
        kwargs["timeout"] = timeout

    if seed is not None:
        kwargs["random_seed"] = seed + attempt_id

    start_time = time.perf_counter()

    embedding = minorminer.find_embedding(
        source_graph,
        target_graph,
        **kwargs,
    )

    runtime = time.perf_counter() - start_time

    if not embedding:
        return {
            "attempt_id": attempt_id,
            "embedding_found": False,
            "runtime": runtime,
            "embedding": None,
            "num_physical_qubits": None,
            "num_logical_variables_embedded": None,
            "max_chain_length": None,
            "min_chain_length": None,
            "mean_chain_length": None,
            "chain_lengths": None,
        }

    chain_lengths = [len(chain) for chain in embedding.values()]
    num_physical_qubits = sum(chain_lengths)
    num_logical_variables_embedded = len(chain_lengths)

    return {
        "attempt_id": attempt_id,
        "embedding_found": True,
        "runtime": runtime,
        "embedding": embedding,
        "num_physical_qubits": num_physical_qubits,
        "num_logical_variables_embedded": num_logical_variables_embedded,
        "max_chain_length": max(chain_lengths) if chain_lengths else 0,
        "min_chain_length": min(chain_lengths) if chain_lengths else 0,
        "mean_chain_length": (
            num_physical_qubits / num_logical_variables_embedded
            if num_logical_variables_embedded > 0
            else 0.0
        ),
        "chain_lengths": chain_lengths,
    }


@dataclass
class EmbeddingFeasibilityConfig:
    """
    Configuration for the embedding-feasibility simulator.

    Parameters
    ----------
    topology:
        Target hardware topology on which the logical QUBO graph is embedded.

        Supported values:
            - "pegasus"
            - "zephyr"
            - "chimera"

    topology_size:
        Size parameter of the target hardware graph.

    num_tries:
        Number of independent minorminer attempts.

    timeout:
        Optional timeout, in seconds, for each minorminer call.

    seed:
        Optional base seed. If provided, each attempt uses seed + attempt_id.

    store_embedding:
        Whether to store the full best embedding dictionary in the metadata.

    num_workers:
        Number of worker processes used to parallelize embedding attempts.

        If None, the simulator uses:

            min(num_tries, os.cpu_count())

        Set num_workers=1 to disable multiprocessing.
    """

    topology: Literal["pegasus", "zephyr", "chimera"] = "pegasus"
    topology_size: int = 16
    num_tries: int = 10
    timeout: int | None = None
    seed: int | None = None
    store_embedding: bool = False
    num_workers: int | None = None


class EmbeddingFeasibilitySimulator(BaseSimulator):
    """
    Simulator-like backend used to test minor-embedding feasibility.

    This backend does not perform annealing. It tries to minor-embed the
    logical QUBO graph onto an ideal D-Wave-like hardware graph.

    The simulator can optionally return oracle solutions from the current
    running problem. These oracle solutions are returned as QUBO variable
    indices, not as original problem elements, because the quantum algorithms
    usually parse samples through their var_to_element map.
    """

    def __init__(self, config: EmbeddingFeasibilityConfig | None = None):
        """
        Initialize the embedding-feasibility backend.

        The target hardware graph is built once and reused for all QUBOs tested
        by this simulator instance.
        """

        self.config = config or EmbeddingFeasibilityConfig()
        self.target_graph = self._build_target_graph()
        self.running_problem = None

    def set_running_problem(self, problem: Any) -> None:
        """
        Store the current problem instance.

        This is used only for oracle mode, where the simulator returns true
        MHS samples so that algorithms whose control flow depends on finding
        a solution can continue running.
        """

        self.running_problem = problem

    def simulate(self, qubo: dict[tuple[int, int], float]) -> SimulatorResultSet:
        """
        Try to minor-embed the logical QUBO graph into the target hardware graph.

        The QUBO is converted to a BQM, its logical graph is extracted, and
        multiple independent minorminer attempts are run. If num_workers > 1,
        these attempts are run in parallel.

        The returned samples are oracle MHS samples converted from original
        problem elements to QUBO variable indices.
        """

        bqm = dimod.BinaryQuadraticModel.from_qubo(qubo)

        start_time = time.perf_counter()

        source_graph = self._build_source_graph(bqm)

        attempt_results = self._run_embedding_attempts(source_graph)

        best_embedding = None
        best_score = None
        successful_tries = 0
        attempt_metadata = []

        for attempt_result in attempt_results:
            if not attempt_result["embedding_found"]:
                attempt_metadata.append(
                    {
                        "attempt_id": attempt_result["attempt_id"],
                        "embedding_found": False,
                        "runtime": attempt_result["runtime"],
                        "num_physical_qubits": None,
                        "max_chain_length": None,
                        "mean_chain_length": None,
                    }
                )
                continue

            successful_tries += 1

            attempt_metadata.append(
                {
                    "attempt_id": attempt_result["attempt_id"],
                    "embedding_found": True,
                    "runtime": attempt_result["runtime"],
                    "num_physical_qubits": attempt_result["num_physical_qubits"],
                    "max_chain_length": attempt_result["max_chain_length"],
                    "mean_chain_length": attempt_result["mean_chain_length"],
                }
            )

            score = (
                attempt_result["num_physical_qubits"],
                attempt_result["max_chain_length"],
            )

            if best_score is None or score < best_score:
                best_score = score
                best_embedding = attempt_result["embedding"]

        total_runtime = time.perf_counter() - start_time

        metadata = self._build_metadata(
            bqm=bqm,
            source_graph=source_graph,
            best_embedding=best_embedding,
            successful_tries=successful_tries,
            attempt_metadata=attempt_metadata,
            total_runtime=total_runtime,
        )

        results = self._build_oracle_results()

        return SimulatorResultSet(
            results=results,
            raw_result=None,
            metadata=metadata,
        )

    def _run_embedding_attempts(
        self,
        source_graph: nx.Graph,
    ) -> list[dict[str, Any]]:
        """
        Run all minorminer attempts, either sequentially or in parallel.

        Attempts are independent, so they can be distributed across multiple
        processes.
        """

        num_workers = self._get_num_workers()

        if num_workers <= 1 or self.config.num_tries <= 1:
            return [
                _run_minorminer_attempt_worker(
                    source_graph=source_graph,
                    target_graph=self.target_graph,
                    attempt_id=attempt_id,
                    timeout=self.config.timeout,
                    seed=self.config.seed,
                )
                for attempt_id in range(self.config.num_tries)
            ]

        results: list[dict[str, Any]] = []

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    _run_minorminer_attempt_worker,
                    source_graph,
                    self.target_graph,
                    attempt_id,
                    self.config.timeout,
                    self.config.seed,
                )
                for attempt_id in range(self.config.num_tries)
            ]

            for future in as_completed(futures):
                results.append(future.result())

        return sorted(results, key=lambda item: item["attempt_id"])

    def _get_num_workers(self) -> int:
        """
        Determine the number of worker processes used for embedding attempts.

        If the config does not specify num_workers, the number is chosen as the
        minimum between num_tries and the available CPU count.
        """

        if self.config.num_workers is not None:
            return max(1, self.config.num_workers)

        cpu_count = os.cpu_count() or 1

        return max(1, min(self.config.num_tries, cpu_count))

    def get_set_from_sample(self, sample: dict[Any, int]) -> frozenset[Any]:
        """
        Convert a binary sample into a selected-set representation.

        This method is included for interface compatibility with other
        simulators.
        """

        return frozenset(variable for variable, value in sample.items() if value == 1)

    def _build_oracle_results(self) -> list[SimulatorResult]:
        """
        Build oracle SimulatorResult objects from the true MHS of the running
        problem.

        The samples returned here are sets of QUBO variable indices, not sets
        of original problem elements.

        This is necessary because the quantum algorithms usually parse solver
        samples by applying their own var_to_element map.
        """

        if self.running_problem is None:
            return []

        if not hasattr(self.running_problem, "minimal_hitting_sets"):
            return []

        element_to_var = self._build_element_to_var_map()

        results: list[SimulatorResult] = []

        for mhs in self.running_problem.minimal_hitting_sets:
            variable_sample = frozenset(
                element_to_var[element]
                for element in mhs
                if element in element_to_var
            )

            results.append(
                SimulatorResult(
                    sample=variable_sample,
                    energy=-np.inf,
                    num_occurrences=1,
                )
            )

        return results

    def _build_element_to_var_map(self) -> dict[Any, int]:
        """
        Build the map from original problem elements to QUBO variable indices.

        This must match the ordering used by the algorithms when they construct
        the QUBO. The expected convention is:

            universe = sorted(problem.universe)
            element_to_var = {element: index for index, element in enumerate(universe)}

        If the problem has no explicit universe attribute, a fallback based on
        the union of test cases is used.
        """

        universe = self._get_ordered_problem_universe()

        return {
            element: index
            for index, element in enumerate(universe)
        }

    def _get_ordered_problem_universe(self) -> list[Any]:
        """
        Return the ordered universe of the current problem.

        The preferred source is running_problem.universe. If unavailable, the
        universe is reconstructed as the sorted union of all test cases.
        """

        if self.running_problem is None:
            return []

        if hasattr(self.running_problem, "universe"):
            return sorted(self.running_problem.universe)

        if hasattr(self.running_problem, "test_cases"):
            universe = set()

            for test_case in self.running_problem.test_cases:
                universe.update(test_case)

            return sorted(universe)

        if hasattr(self.running_problem, "size"):
            return list(range(self.running_problem.size))

        raise AttributeError(
            "Cannot infer the problem universe. Expected the problem to have "
            "one of: universe, test_cases, or size."
        )

    def _build_target_graph(self) -> nx.Graph:
        """
        Build the ideal hardware graph used as the embedding target.
        """

        if self.config.topology == "pegasus":
            return dnx.pegasus_graph(self.config.topology_size)

        if self.config.topology == "zephyr":
            return dnx.zephyr_graph(self.config.topology_size)

        if self.config.topology == "chimera":
            return dnx.chimera_graph(self.config.topology_size)

        raise ValueError(f"Unsupported topology: {self.config.topology}")

    def _build_source_graph(self, bqm: dimod.BinaryQuadraticModel) -> nx.Graph:
        """
        Build the logical graph induced by the QUBO/BQM.

        Logical variables become nodes. Non-zero quadratic interactions become
        edges. Linear terms are represented only as nodes.
        """

        source_graph = nx.Graph()
        source_graph.add_nodes_from(bqm.variables)

        for u, v, bias in bqm.iter_quadratic():
            if bias != 0:
                source_graph.add_edge(u, v)

        return source_graph

    def _compute_embedding_stats(
        self,
        embedding: dict[Any, list[Any]],
    ) -> dict[str, Any]:
        """
        Compute summary statistics for a valid embedding.
        """

        chain_lengths = [len(chain) for chain in embedding.values()]

        num_physical_qubits = sum(chain_lengths)
        num_logical_variables_embedded = len(chain_lengths)

        return {
            "num_physical_qubits": num_physical_qubits,
            "num_logical_variables_embedded": num_logical_variables_embedded,
            "max_chain_length": max(chain_lengths) if chain_lengths else 0,
            "min_chain_length": min(chain_lengths) if chain_lengths else 0,
            "mean_chain_length": (
                num_physical_qubits / num_logical_variables_embedded
                if num_logical_variables_embedded > 0
                else 0.0
            ),
            "chain_lengths": chain_lengths,
        }

    def _build_metadata(
        self,
        bqm: dimod.BinaryQuadraticModel,
        source_graph: nx.Graph,
        best_embedding: dict[Any, list[Any]] | None,
        successful_tries: int,
        attempt_metadata: list[dict[str, Any]],
        total_runtime: float,
    ) -> dict[str, Any]:
        """
        Build the metadata dictionary returned by the simulator.
        """

        metadata = {
            "simulator_type": "embedding_feasibility",
            "topology": self.config.topology,
            "topology_size": self.config.topology_size,
            "num_tries": self.config.num_tries,
            "num_workers": self._get_num_workers(),
            "timeout": self.config.timeout,
            "seed": self.config.seed,
            "runtime": total_runtime,
            "successful_tries": successful_tries,
            "embedding_found": best_embedding is not None,
            "num_logical_variables": len(bqm.variables),
            "num_logical_couplers": len(bqm.quadratic),
            "num_source_nodes": source_graph.number_of_nodes(),
            "num_source_edges": source_graph.number_of_edges(),
            "target_num_nodes": self.target_graph.number_of_nodes(),
            "target_num_edges": self.target_graph.number_of_edges(),
            "attempts": attempt_metadata,
        }

        if best_embedding is None:
            metadata.update(
                {
                    "num_physical_qubits": None,
                    "num_logical_variables_embedded": None,
                    "max_chain_length": None,
                    "min_chain_length": None,
                    "mean_chain_length": None,
                    "physical_logical_qubit_ratio": None,
                    "chain_lengths": None,
                    "embedding": None,
                }
            )

            return metadata

        embedding_stats = self._compute_embedding_stats(best_embedding)

        num_logical_variables = len(bqm.variables)
        num_physical_qubits = embedding_stats["num_physical_qubits"]

        metadata.update(
            {
                "num_physical_qubits": num_physical_qubits,
                "num_logical_variables_embedded": embedding_stats[
                    "num_logical_variables_embedded"
                ],
                "max_chain_length": embedding_stats["max_chain_length"],
                "min_chain_length": embedding_stats["min_chain_length"],
                "mean_chain_length": embedding_stats["mean_chain_length"],
                "physical_logical_qubit_ratio": (
                    num_physical_qubits / num_logical_variables
                    if num_logical_variables > 0
                    else 0.0
                ),
                "chain_lengths": embedding_stats["chain_lengths"],
                "embedding": best_embedding if self.config.store_embedding else None,
            }
        )

        return metadata