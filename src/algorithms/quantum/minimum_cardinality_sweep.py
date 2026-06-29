from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import time

from src.algorithms.quantum.base_algorithm import BaseAlgorithm
from src.simulator.base_simulator import BaseSimulator
from src.core.result import AlgorithmResult, AlgorithmResultSet
from src.core.problem import HittingSetProblem
from src.utils.validation import is_hitting_set

from src.preprocessing.preprocessing_utils import (
    preprocess_hitting_set_instance,
    remove_dominated_test_cases,
)


class MinimumSearchMode(str, Enum):
    """
    Search strategy over the target cardinality K.

    LINEAR_SWEEP:
        Test K in increasing order and stop at the first feasible layer.

    BINARY_SEARCH:
        Use the monotonicity of the hitting-set decision problem:
        if a hitting set of size K exists, then a hitting set of size K' > K
        also exists.
    """

    LINEAR_SWEEP = "linear_sweep"
    BINARY_SEARCH = "binary_search"


@dataclass
class MinimumPUBOConfig:
    """
    Configuration for the PUBO/QUBO algorithm that searches for minimum
    hitting sets.

    Parameters
    ----------
    scale:
        Weight of the coverage constraint terms.

    cardinality_scale:
        Weight of the cardinality penalty.

    normalize:
        If True, divide all QUBO coefficients by the largest absolute
        coefficient.

    prune_test_cases:
        If True, remove redundant test cases before constructing the QUBO.

    use_lower_bound:
        If True, compute classical lower bounds and start the cardinality
        search from the strongest lower bound instead of starting from K = 1.

    use_lp_lower_bound:
        If True, include the LP relaxation lower bound.

    use_greedy_upper_bound:
        If True, compute a greedy hitting set and use its size as an upper
        bound for the cardinality search.

    use_cardinality_limited_coverage_ancillas:
        If True, reduce the number of coverage ancillas using the fact that
        the current QUBO is constructed at fixed cardinality K.

    search_mode:
        Strategy used to choose the next tested cardinality.
    """

    scale: float = 1.0
    cardinality_scale: float = 1.0
    normalize: bool = False
    prune_test_cases: bool = False

    use_lower_bound: bool = True
    use_lp_lower_bound: bool = False
    use_greedy_upper_bound: bool = True

    use_cardinality_limited_coverage_ancillas: bool = False

    search_mode: MinimumSearchMode = MinimumSearchMode.LINEAR_SWEEP


@dataclass
class MinimumPUBOResultSet(AlgorithmResultSet):
    """
    Result set for the minimum hitting set PUBO algorithm.

    The benchmark code often asks for the total number of reads used at a
    given cardinality. This method extracts it from the metadata.

    The num_qubits and num_couplers fields store the maximum logical size
    reached across all QUBOs generated during the cardinality search.
    """

    def get_total_count(self, length):
        return self.metadata[length]["simulator_metadata"]["num_reads"]


class MinimumPUBOAlgorithm(BaseAlgorithm):
    """
    PUBO/QUBO algorithm for finding minimum hitting sets.

    The QUBO construction is based on the fixed-cardinality formulation. The
    algorithm then searches over cardinalities K using one of two strategies:

    1. Linear sweep.
    2. Binary search.

    The formulation is the same in both cases. Only the cardinality scheduler
    changes.

    Important
    ---------
    This class is for the minimum hitting set problem, not for enumerating all
    minimal hitting sets.

    Therefore, unlike the recursive minimal algorithm, this class does not add
    penalties against previously found minimal hitting sets.
    """

    mode: str = "minimum"

    def __init__(self, config: MinimumPUBOConfig | None = None):
        self.config = config or MinimumPUBOConfig()

        # If normalize=True, this stores the largest coefficient used to
        # rescale the QUBO. It is needed to compare sampled energies with the
        # expected ground-state energy.
        self.normalization_constant = 1.0

    # ------------------------------------------------------------------
    # Preprocessing utilities
    # ------------------------------------------------------------------

    def prepare_test_cases(
        self,
        test_cases: List[Set[int]],
    ) -> List[Set[int]]:
        """
        Prepare the test cases used internally by the algorithm.

        This method never mutates the original problem object.
        """
        prepared = [set(test_case) for test_case in test_cases]

        if self.config.prune_test_cases:
            prepared = remove_dominated_test_cases(prepared)

        return prepared

    def compute_search_range(
        self,
        universe: List[int],
        test_cases: List[Set[int]],
    ) -> tuple[int, int, dict]:
        """
        Compute the cardinality range for the search.

        Returns
        -------
        tuple[int, int, dict]
            start_cardinality:
                First cardinality to test.

            stop_cardinality:
                Last cardinality to test.

            preprocessing_metadata:
                Dictionary containing lower bounds, greedy upper bound,
                and preprocessing information.
        """
        max_possible_cardinality = min(len(universe), len(test_cases))

        # If there are no test cases, the empty set is the minimum hitting set.
        if len(test_cases) == 0:
            metadata = {
                "empty_instance": True,
                "start_cardinality": 0,
                "stop_cardinality": 0,
            }
            return 0, 0, metadata

        preprocessing = preprocess_hitting_set_instance(
            universe=universe,
            test_cases=test_cases,
            use_lp_lower_bound=self.config.use_lp_lower_bound,
            prune_dominated=False,
        )

        preprocessing_metadata = preprocessing.as_dict()
        preprocessing_metadata["empty_instance"] = False
        preprocessing_metadata["use_lower_bound"] = self.config.use_lower_bound
        preprocessing_metadata["use_lp_lower_bound"] = self.config.use_lp_lower_bound
        preprocessing_metadata["use_greedy_upper_bound"] = self.config.use_greedy_upper_bound

        if self.config.use_lower_bound:
            start_cardinality = preprocessing.start_cardinality
        else:
            start_cardinality = 1

        if self.config.use_greedy_upper_bound:
            stop_cardinality = preprocessing.stop_cardinality
        else:
            stop_cardinality = max_possible_cardinality

        # Safety clipping.
        start_cardinality = max(1, start_cardinality)
        stop_cardinality = min(stop_cardinality, max_possible_cardinality)

        preprocessing_metadata["start_cardinality_used"] = start_cardinality
        preprocessing_metadata["stop_cardinality_used"] = stop_cardinality

        return start_cardinality, stop_cardinality, preprocessing_metadata

    def find_length_one_solutions(
        self,
        test_cases: List[Set[int]],
    ) -> Set[int]:
        """
        Find all hitting sets of cardinality 1.

        An element is a size-1 hitting set if it belongs to every test case.
        """
        if len(test_cases) == 0:
            return set()

        return set.intersection(*test_cases)

    # ------------------------------------------------------------------
    # Cardinality search policy
    # ------------------------------------------------------------------

    def initialize_cardinality_search(
        self,
        start_cardinality: int,
        stop_cardinality: int,
    ) -> Dict[str, Any]:
        """
        Initialize the cardinality search state.

        The main run() loop does not need to know whether the algorithm is
        doing linear sweep or binary search. This dictionary stores the state
        needed by the search scheduler.
        """
        first_qubo_cardinality = max(2, start_cardinality)

        return {
            "mode": self.config.search_mode,
            "start_cardinality": start_cardinality,
            "stop_cardinality": stop_cardinality,
            "current": first_qubo_cardinality,
            "left": first_qubo_cardinality,
            "right": stop_cardinality,
            "best_cardinality": None,
            "tested_cardinalities": [],
            "done": first_qubo_cardinality > stop_cardinality,
        }

    def get_next_cardinality(
        self,
        search_state: Dict[str, Any],
    ) -> Optional[int]:
        """
        Return the next cardinality K to test.

        The QUBO construction and sampling logic remain independent of the
        search mode.
        """
        if search_state["done"]:
            return None

        mode = search_state["mode"]

        if mode == MinimumSearchMode.LINEAR_SWEEP:
            current = search_state["current"]
            stop = search_state["stop_cardinality"]

            if current > stop:
                search_state["done"] = True
                return None

            return current

        if mode == MinimumSearchMode.BINARY_SEARCH:
            left = search_state["left"]
            right = search_state["right"]

            if left > right:
                search_state["done"] = True
                return None

            return (left + right) // 2

        raise ValueError(f"Unknown minimum search mode: {mode}")

    def update_cardinality_search(
        self,
        search_state: Dict[str, Any],
        cardinality: int,
        found_solution: bool,
    ) -> None:
        """
        Update the cardinality search state after testing one cardinality.
        """
        mode = search_state["mode"]

        search_state["tested_cardinalities"].append(cardinality)

        if mode == MinimumSearchMode.LINEAR_SWEEP:
            if found_solution:
                search_state["best_cardinality"] = cardinality
                search_state["done"] = True
            else:
                search_state["current"] = cardinality + 1

                if search_state["current"] > search_state["stop_cardinality"]:
                    search_state["done"] = True

            return

        if mode == MinimumSearchMode.BINARY_SEARCH:
            if found_solution:
                search_state["best_cardinality"] = cardinality
                search_state["right"] = cardinality - 1
            else:
                search_state["left"] = cardinality + 1

            if search_state["left"] > search_state["right"]:
                search_state["done"] = True

            return

        raise ValueError(f"Unknown minimum search mode: {mode}")

    # ------------------------------------------------------------------
    # QUBO construction
    # ------------------------------------------------------------------

    def get_num_coverage_ancillas(
        self,
        test_case_size: int,
        cardinality: int,
    ) -> int:
        """
        Compute the number of ancilla bits needed for one coverage constraint.

        For a conflict C_j, define:

            s_j = |C_j cap X|

        The coverage constraint uses:

            (s_j - t_j - 1)^2

        so the auxiliary integer t_j represents:

            t_j = s_j - 1

        If cardinality-limited coverage ancillas are enabled, then the current
        QUBO is built at fixed cardinality K, so:

            |C_j cap X| <= min(|C_j|, K)

        and fewer ancillas may be sufficient.
        """
        if test_case_size <= 1:
            return 0

        if self.config.use_cardinality_limited_coverage_ancillas:
            max_overlap = min(test_case_size, cardinality)
        else:
            max_overlap = test_case_size

        if max_overlap <= 1:
            return 0

        return int(np.ceil(np.log2(max_overlap)))

    def construct_hamiltonian(
        self,
        universe: List[int],
        test_cases: List[Set[int]],
        cardinality: int,
    ) -> Dict[Tuple[int, int], float]:
        """
        Construct the QUBO matrix for the minimum hitting set problem.

        The Hamiltonian has two parts:

        1. Coverage constraints.

            For each test case C_j:

                s_j = sum_{i in C_j} x_i

            The constraint s_j >= 1 is encoded with ancillas as:

                (s_j - t_j - 1)^2

        2. Cardinality penalty.

                (sum_i x_i - K)^2
        """
        self.normalization_constant = 1.0

        Q: Dict[Tuple[int, int], float] = {}

        universe_set = set(universe)

        assert all(test_case.issubset(universe_set) for test_case in test_cases), (
            "The test cases contain elements outside the universe."
        )

        ordered_universe = sorted(universe)

        element_to_var = {
            element: idx
            for idx, element in enumerate(ordered_universe)
        }

        num_problem_vars = len(ordered_universe)

        # ------------------------------------------------------------
        # Ancilla allocation for test-case coverage constraints.
        # ------------------------------------------------------------
        test_cases_offset = []
        next_free_qubit = num_problem_vars

        for test_case in test_cases:
            test_cases_offset.append(next_free_qubit)

            num_ancilla = self.get_num_coverage_ancillas(
                test_case_size=len(test_case),
                cardinality=cardinality,
            )

            next_free_qubit += num_ancilla

        # ------------------------------------------------------------
        # Coverage constraints.
        #
        # For each test case C_j:
        #
        #     s_j = sum_{i in C_j} x_i
        #     t_j = sum_k 2^k y_{j,k}
        #
        # Add:
        #
        #     scale * (s_j - t_j - 1)^2
        #
        # The constant +1 is omitted from the QUBO dictionary and accounted
        # for in ground_state_energy().
        # ------------------------------------------------------------
        for j, test_case in enumerate(test_cases):
            ancilla_offset = test_cases_offset[j]

            num_ancilla = self.get_num_coverage_ancillas(
                test_case_size=len(test_case),
                cardinality=cardinality,
            )

            tc_vars = sorted(element_to_var[element] for element in test_case)

            # t_j^2
            for k in range(num_ancilla):
                yk = ancilla_offset + k

                Q[(yk, yk)] = Q.get((yk, yk), 0.0) + (
                    2.0 ** (2 * k)
                ) * self.config.scale

                for k_prime in range(k + 1, num_ancilla):
                    ykp = ancilla_offset + k_prime

                    Q[(yk, ykp)] = Q.get((yk, ykp), 0.0) + (
                        2.0 ** (k + k_prime + 1)
                    ) * self.config.scale

            # + 2 t_j
            for k in range(num_ancilla):
                yk = ancilla_offset + k

                Q[(yk, yk)] = Q.get((yk, yk), 0.0) + (
                    2.0 * (2.0 ** k)
                ) * self.config.scale

            # s_j^2
            for idx, i in enumerate(tc_vars):
                Q[(i, i)] = Q.get((i, i), 0.0) + self.config.scale

                for i_prime in tc_vars[idx + 1:]:
                    Q[(i, i_prime)] = Q.get((i, i_prime), 0.0) + (
                        2.0 * self.config.scale
                    )

            # - 2 s_j
            for i in tc_vars:
                Q[(i, i)] = Q.get((i, i), 0.0) - (
                    2.0 * self.config.scale
                )

            # - 2 s_j t_j
            for i in tc_vars:
                for k in range(num_ancilla):
                    yk = ancilla_offset + k

                    Q[(i, yk)] = Q.get((i, yk), 0.0) - (
                        2.0 * (2.0 ** k) * self.config.scale
                    )

        # ------------------------------------------------------------
        # Cardinality penalty:
        #
        #     cardinality_scale * (sum_i x_i - K)^2
        #
        # The constant K^2 is omitted from the QUBO dictionary and accounted
        # for in ground_state_energy().
        # ------------------------------------------------------------
        for i in range(num_problem_vars):
            Q[(i, i)] = Q.get((i, i), 0.0) + (
                1.0 - 2.0 * cardinality
            ) * self.config.cardinality_scale

            for i_prime in range(i + 1, num_problem_vars):
                Q[(i, i_prime)] = Q.get((i, i_prime), 0.0) + (
                    2.0 * self.config.cardinality_scale
                )

        # ------------------------------------------------------------
        # Optional coefficient normalization.
        # ------------------------------------------------------------
        if self.config.normalize and len(Q) > 0:
            max_abs_coefficient = max(abs(value) for value in Q.values())

            if max_abs_coefficient > 0:
                self.normalization_constant = max_abs_coefficient

                for key in list(Q.keys()):
                    Q[key] /= self.normalization_constant

        return Q

    # ------------------------------------------------------------------
    # Candidate parsing, energy handling, and logical size utilities
    # ------------------------------------------------------------------

    def ground_state_energy(
        self,
        num_test_cases: int,
        cardinality: int,
    ) -> float:
        """
        Compute the expected ground-state energy for valid hitting sets of
        the target cardinality.

        The omitted constants are not included in the QUBO, so for a valid
        hitting set of cardinality K the variable-dependent energy is:

            - num_test_cases * scale - K^2 * cardinality_scale
        """
        energy = (
            -num_test_cases * self.config.scale
            -cardinality * cardinality * self.config.cardinality_scale
        )

        return energy / self.normalization_constant

    def parse_candidate_sample(
        self,
        sample: Set[int],
        var_to_element: Dict[int, int],
        num_problem_vars: int,
    ) -> frozenset[int]:
        """
        Convert a sampled qubit configuration into a set of original elements.

        Ancilla variables are ignored.
        """
        parsed = [
            var_to_element[qubit]
            for qubit in sample
            if qubit < num_problem_vars
        ]

        return frozenset(parsed)

    def get_num_qubits_from_qubo(
        self,
        Q: Dict[Tuple[int, int], float],
    ) -> int:
        """
        Compute the number of logical qubits used by a QUBO dictionary.
        """
        if len(Q) == 0:
            return 0

        return max(max(i, j) for i, j in Q.keys()) + 1

    def get_num_couplers_from_qubo(
        self,
        Q: Dict[Tuple[int, int], float],
    ) -> int:
        """
        Compute the number of logical couplers used by a QUBO dictionary.

        A logical coupler is a non-zero off-diagonal QUBO term Q_{ij}
        with i != j. Diagonal terms Q_{ii} are local biases and are not
        counted as couplers.
        """
        return sum(
            1
            for (i, j), value in Q.items()
            if i != j and value != 0.0
        )

    # ------------------------------------------------------------------
    # Main algorithm
    # ------------------------------------------------------------------

    def run(
        self,
        problem: HittingSetProblem,
        simulator: BaseSimulator,
    ) -> AlgorithmResultSet:
        """
        Run the minimum hitting set PUBO algorithm.

        The QUBO construction and validation logic are shared by both linear
        sweep and binary search. The search mode only changes the sequence of
        tested cardinalities.
        """
        mhs_found: List[AlgorithmResult] = []
        algorithm_metadata: Dict[Any, Any] = {}

        ordered_universe = sorted(problem.universe)

        var_to_element = {
            idx: element
            for idx, element in enumerate(ordered_universe)
        }

        num_problem_vars = len(ordered_universe)

        # Work on a local copy of the test cases.
        # Do not mutate problem.test_cases.
        test_cases = self.prepare_test_cases(problem.test_cases)

        # ------------------------------------------------------------
        # Empty instance: the empty set is the minimum hitting set.
        # ------------------------------------------------------------
        if len(test_cases) == 0:
            mhs_found.append(AlgorithmResult(frozenset(), 1))

            algorithm_metadata["preprocessing"] = {
                "empty_instance": True,
                "start_cardinality_used": 0,
                "stop_cardinality_used": 0,
            }

            algorithm_metadata["search"] = {
                "mode": self.config.search_mode.value,
                "tested_cardinalities": [],
                "best_cardinality": 0,
            }

            return MinimumPUBOResultSet(
                problem=problem,
                mode=self.mode,
                num_qubits=0,
                num_couplers=0,
                solutions=mhs_found,
                metadata=algorithm_metadata,
            )

        # ------------------------------------------------------------
        # Compute lower bounds and optional greedy upper bound.
        # ------------------------------------------------------------
        start_cardinality, stop_cardinality, preprocessing_metadata = (
            self.compute_search_range(
                universe=ordered_universe,
                test_cases=test_cases,
            )
        )

        algorithm_metadata["preprocessing"] = {
            **preprocessing_metadata,
            "num_test_cases_original": len(problem.test_cases),
            "num_test_cases_used": len(test_cases),
        }

        max_num_qubits = 0
        max_num_couplers = 0

        # ------------------------------------------------------------
        # Directly handle cardinality 1.
        #
        # If there is a length-1 hitting set, it is immediately optimal.
        # No QUBO call is needed.
        #
        # If there is no length-1 hitting set, then K = 1 is known to be
        # infeasible and the QUBO-based search can safely start from K = 2.
        # ------------------------------------------------------------
        if start_cardinality <= 1:
            len_1_solutions = self.find_length_one_solutions(test_cases)

            if len(len_1_solutions) > 0:
                for sol in len_1_solutions:
                    mhs_found.append(
                        AlgorithmResult(
                            set=frozenset([sol]),
                            num_occurrences=1,
                        )
                    )

                algorithm_metadata[1] = {
                    "direct_solution": True,
                    "matrix_construction_time": 0.0,
                    "num_qubits": 0,
                    "num_couplers": 0,
                    "simulator_metadata": {
                        "num_reads": 1,
                    },
                    "found_solution": True,
                    "num_valid_solutions_sampled": len(len_1_solutions),
                }

                algorithm_metadata["search"] = {
                    "mode": self.config.search_mode.value,
                    "start_cardinality": start_cardinality,
                    "stop_cardinality": stop_cardinality,
                    "tested_cardinalities": [1],
                    "best_cardinality": 1,
                }

                return MinimumPUBOResultSet(
                    problem=problem,
                    mode=self.mode,
                    num_qubits=0,
                    num_couplers=0,
                    solutions=mhs_found,
                    metadata=algorithm_metadata,
                )

        # ------------------------------------------------------------
        # Shared cardinality-search loop.
        #
        # The formulation is always the same fixed-cardinality QUBO.
        # The search_state decides which K is tested next.
        # ------------------------------------------------------------
        best_mhs_counts: Dict[frozenset[int], int] = {}

        search_state = self.initialize_cardinality_search(
            start_cardinality=start_cardinality,
            stop_cardinality=stop_cardinality,
        )

        while True:
            cardinality = self.get_next_cardinality(search_state)

            if cardinality is None:
                break

            iteration_metadata: Dict[str, Any] = {}
            mhs_counts: Dict[frozenset[int], int] = {}

            # --------------------------------------------------------
            # Build QUBO.
            # --------------------------------------------------------
            start_matrix = time.time()

            Q_matrix = self.construct_hamiltonian(
                universe=ordered_universe,
                test_cases=test_cases,
                cardinality=cardinality,
            )

            end_matrix = time.time()

            iteration_metadata["matrix_construction_time"] = (
                end_matrix - start_matrix
            )

            current_num_qubits = self.get_num_qubits_from_qubo(Q_matrix)
            current_num_couplers = self.get_num_couplers_from_qubo(Q_matrix)

            max_num_qubits = max(max_num_qubits, current_num_qubits)
            max_num_couplers = max(max_num_couplers, current_num_couplers)

            iteration_metadata["num_qubits"] = current_num_qubits
            iteration_metadata["num_couplers"] = current_num_couplers

            # --------------------------------------------------------
            # Run simulator.
            # --------------------------------------------------------
            candidates = simulator.simulate(Q_matrix)

            iteration_metadata["simulator_metadata"] = candidates.metadata

            # --------------------------------------------------------
            # Filter only sampled ground states.
            # --------------------------------------------------------
            gs_energy = self.ground_state_energy(
                num_test_cases=len(test_cases),
                cardinality=cardinality,
            )

            tol = 1.0e-8

            for candidate in candidates.results:
                if candidate.energy > gs_energy + tol:
                    continue

                parsed_result = self.parse_candidate_sample(
                    sample=candidate.sample,
                    var_to_element=var_to_element,
                    num_problem_vars=num_problem_vars,
                )

                # At fixed K, for the minimum problem we only need a hitting
                # set of size K.
                #
                # Do not require minimality here. In binary search, the tested
                # cardinality may be larger than the true optimum, so valid
                # feasible sets may be non-minimal supersets.
                if len(parsed_result) != cardinality:
                    continue

                if is_hitting_set(parsed_result, test_cases):
                    mhs_counts[parsed_result] = (
                        mhs_counts.get(parsed_result, 0)
                        + candidate.num_occurrences
                    )

            found_solution = len(mhs_counts) > 0

            iteration_metadata["found_solution"] = found_solution
            iteration_metadata["num_valid_solutions_sampled"] = len(mhs_counts)

            algorithm_metadata[cardinality] = iteration_metadata

            # --------------------------------------------------------
            # Store best feasible layer found so far.
            #
            # Linear sweep:
            #     the first feasible layer is the optimum found by the method.
            #
            # Binary search:
            #     every feasible K becomes the current best, then the search
            #     continues below K.
            # --------------------------------------------------------
            if found_solution:
                best_mhs_counts = mhs_counts

            self.update_cardinality_search(
                search_state=search_state,
                cardinality=cardinality,
                found_solution=found_solution,
            )

        for mhs, count in best_mhs_counts.items():
            mhs_found.append(
                AlgorithmResult(
                    set=mhs,
                    num_occurrences=count,
                )
            )

        algorithm_metadata["search"] = {
            "mode": self.config.search_mode.value,
            "start_cardinality": start_cardinality,
            "stop_cardinality": stop_cardinality,
            "tested_cardinalities": search_state["tested_cardinalities"],
            "best_cardinality": search_state["best_cardinality"],
        }

        # ------------------------------------------------------------
        # Important:
        #
        # num_qubits and num_couplers are the maximum logical sizes used by
        # any QUBO built during the cardinality search, not merely the logical
        # sizes of the last tested cardinality.
        # ------------------------------------------------------------
        return MinimumPUBOResultSet(
            problem=problem,
            mode=self.mode,
            num_qubits=max_num_qubits,
            num_couplers=max_num_couplers,
            solutions=mhs_found,
            metadata=algorithm_metadata,
        )