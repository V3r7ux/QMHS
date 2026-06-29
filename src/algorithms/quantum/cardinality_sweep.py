from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List, Set
import numpy as np
import time

from src.algorithms.quantum.base_algorithm import BaseAlgorithm
from src.simulator.base_simulator import BaseSimulator
from src.core.result import AlgorithmResult, AlgorithmResultSet
from src.core.problem import HittingSetProblem
from src.utils.validation import is_minimal_hitting_set

from src.preprocessing.preprocessing_utils import (
    preprocess_hitting_set_instance,
    remove_dominated_test_cases,
    prune_length_one_solutions,
)


@dataclass
class MinimalCardinalitySweepConfig:
    """
    Configuration for the cardinality-sweep QUBO algorithm used to sample
    minimal hitting sets.

    Parameters
    ----------
    scale:
        Weight of the coverage constraints.

        These terms enforce that every test case/conflict is hit at least once.

    cardinality_scale:
        Weight of the cardinality constraint.

        For each target cardinality K, the algorithm adds:

            (sum_i x_i - K)^2

        to force the sampler to search inside the K-cardinality layer.

    normalize:
        If True, divide all QUBO coefficients by the largest absolute
        coefficient.

        The normalization constant is stored internally so that the expected
        ground-state energy can be rescaled consistently.

    prune_test_cases:
        If True, remove duplicate and dominated test cases before constructing
        the QUBO.

        A test case C_big is dominated if there exists C_small such that:

            C_small subseteq C_big

        In that case C_big can be removed safely.

    use_lower_bound:
        If True, compute a classical lower bound and start the cardinality
        sweep from that bound instead of starting from 1.

    use_lp_lower_bound:
        If True, also compute the LP-relaxation lower bound.

        This can give a stronger lower bound, but requires scipy.
    """

    scale: float = 1.0
    cardinality_scale: float = 1.0
    normalize: bool = False
    prune_test_cases: bool = False

    use_lower_bound: bool = True
    use_lp_lower_bound: bool = False


@dataclass
class MinimalCardinalitySweepResultSet(AlgorithmResultSet):
    """
    Result set for the minimal cardinality-sweep algorithm.

    The benchmark code can use get_total_count(length) to recover how many
    reads were used at a given cardinality.

    The num_qubits and num_couplers fields store the maximum logical size
    reached across all cardinalities tested by the sweep.
    """

    def get_total_count(self, length):
        return self.metadata[length]["simulator_metadata"]["num_reads"]


class MinimalCardinalitySweepAlgorithm(BaseAlgorithm):
    """
    Cardinality-sweep QUBO algorithm for sampling minimal hitting sets.

    The algorithm sweeps over cardinalities:

        K = start_cardinality, ..., max_length

    For each K, it constructs a QUBO whose ground states correspond to
    hitting sets of cardinality K. Since the formulation itself does not
    enforce minimality, sampled ground states are post-filtered using
    is_minimal_hitting_set.

    Unlike the recursive PUBO algorithm, this class does not add penalties
    against previously found minimal hitting sets. Therefore, larger
    non-minimal supersets can still appear in the ground-state manifold, but
    they are discarded during post-processing.
    """

    mode: str = "minimal"

    def __init__(self, config: MinimalCardinalitySweepConfig | None = None):
        self.config = config or MinimalCardinalitySweepConfig()

        # Used only when normalize=True.
        self.normalization_constant = 1.0

    # ------------------------------------------------------------------
    # Preprocessing utilities
    # ------------------------------------------------------------------

    def prepare_test_cases(
        self,
        test_cases: List[Set[int]],
    ) -> List[Set[int]]:
        """
        Prepare test cases used internally by the algorithm.

        The original problem object is not modified.

        Parameters
        ----------
        test_cases:
            Original test cases/conflicts.

        Returns
        -------
        List[Set[int]]
            Test cases used internally.
        """
        prepared = [set(test_case) for test_case in test_cases]

        if self.config.prune_test_cases:
            prepared = remove_dominated_test_cases(prepared)

        return prepared

    def compute_start_cardinality(
        self,
        universe: List[int],
        test_cases: List[Set[int]],
    ) -> tuple[int, dict]:
        """
        Compute the first cardinality to test.

        For minimal enumeration, the lower bound is only used to skip
        impossible cardinalities below the minimum hitting-set size.

        We do not use a greedy upper bound as a stopping point, because
        minimal hitting sets can be larger than minimum hitting sets.

        Parameters
        ----------
        universe:
            Sorted universe elements.

        test_cases:
            Test cases after optional pruning.

        Returns
        -------
        tuple[int, dict]
            start_cardinality:
                First cardinality to test.

            preprocessing_metadata:
                Metadata describing the preprocessing step.
        """
        if len(test_cases) == 0:
            metadata = {
                "empty_instance": True,
                "start_cardinality": 0,
                "start_cardinality_used": 0,
            }
            return 0, metadata

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
        preprocessing_metadata["note"] = (
            "For minimal enumeration, the greedy upper bound is stored "
            "for metadata only and is not used as a stopping cardinality."
        )

        if self.config.use_lower_bound:
            start_cardinality = preprocessing.start_cardinality
        else:
            start_cardinality = 1

        start_cardinality = max(1, start_cardinality)

        preprocessing_metadata["start_cardinality_used"] = start_cardinality

        return start_cardinality, preprocessing_metadata

    def find_length_one_solutions(
        self,
        test_cases: List[Set[int]],
    ) -> Set[int]:
        """
        Find all minimal hitting sets of cardinality 1.

        An element is a size-1 hitting set if it belongs to every test case.

        Parameters
        ----------
        test_cases:
            Test cases/conflicts.

        Returns
        -------
        Set[int]
            Elements that alone hit every test case.
        """
        if len(test_cases) == 0:
            return set()

        return set.intersection(*test_cases)

    # ------------------------------------------------------------------
    # QUBO construction
    # ------------------------------------------------------------------

    def construct_hamiltonian(
        self,
        universe: List[int],
        test_cases: List[Set[int]],
        cardinality: int,
    ) -> Dict[Tuple[int, int], float]:
        """
        Construct the QUBO matrix for the cardinality-sweep formulation.

        Parameters
        ----------
        universe:
            List of universe elements.

        test_cases:
            List of conflicts/test cases.

        cardinality:
            Target cardinality K.

        Returns
        -------
        Dict[Tuple[int, int], float]
            QUBO dictionary.

        Hamiltonian structure
        ---------------------
        The QUBO contains:

        1. Coverage constraints.

           For each test case C_j:

               s_j = sum_{i in C_j} x_i
               t_j = sum_k 2^k y_{j,k}

           The penalty is:

               scale * (s_j - t_j - 1)^2

           This is zero whenever C_j is hit at least once.

        2. Cardinality penalty.

               cardinality_scale * (sum_i x_i - K)^2
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
        # Allocate ancillas for coverage constraints.
        # ------------------------------------------------------------
        test_cases_offset: List[int] = []
        next_free_qubit = num_problem_vars

        for test_case in test_cases:
            test_cases_offset.append(next_free_qubit)

            # t_j = s_j - 1 ranges from 0 to |C_j| - 1.
            # Therefore ceil(log2(|C_j|)) bits are enough.
            num_ancilla = int(np.ceil(np.log2(len(test_case))))
            next_free_qubit += num_ancilla

        # ------------------------------------------------------------
        # Coverage constraints.
        # ------------------------------------------------------------
        for j, test_case in enumerate(test_cases):
            ancilla_offset = test_cases_offset[j]
            num_ancilla = int(np.ceil(np.log2(len(test_case))))

            tc_vars = sorted(element_to_var[element] for element in test_case)

            # t_j^2, where t_j = sum_k 2^k y_{j,k}
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
                    2.0 * (2.0 ** k) * self.config.scale
                )

            # s_j^2, where s_j = sum_{i in C_j} x_i
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
        #     (sum_i x_i - cardinality)^2
        #
        # Constant terms are omitted.
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
    # Energy and parsing utilities
    # ------------------------------------------------------------------

    def ground_state_energy(
        self,
        num_test_cases: int,
        cardinality: int,
    ) -> float:
        """
        Compute the expected ground-state energy for a valid hitting set of
        the target cardinality.

        Parameters
        ----------
        num_test_cases:
            Number of test cases used in the QUBO.

        cardinality:
            Target cardinality K.

        Returns
        -------
        float
            Expected ground-state energy after optional normalization.
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
        Convert a sampled qubit configuration into original universe elements.

        Ancilla qubits are ignored.

        Parameters
        ----------
        sample:
            Set of active qubit indices returned by the simulator.

        var_to_element:
            Mapping from compact qubit indices to original elements.

        num_problem_vars:
            Number of real problem variables.

        Returns
        -------
        frozenset[int]
            Candidate set in original universe labels.
        """
        parsed = [
            var_to_element[qubit]
            for qubit in sample
            if qubit < num_problem_vars
        ]

        return frozenset(parsed)

    def count_num_qubits(
        self,
        Q: Dict[Tuple[int, int], float],
        num_problem_vars: int,
    ) -> int:
        """
        Count the number of qubits appearing in the QUBO.

        Parameters
        ----------
        Q:
            QUBO dictionary.

        num_problem_vars:
            Number of real problem variables.

        Returns
        -------
        int
            Number of qubits used.
        """
        if len(Q) == 0:
            return num_problem_vars

        return max(max(pair) for pair in Q.keys()) + 1

    def count_num_couplers(
        self,
        Q: Dict[Tuple[int, int], float],
    ) -> int:
        """
        Count the number of logical couplers appearing in the QUBO.

        A logical coupler is a non-zero off-diagonal QUBO term Q_{ij}
        with i != j. Diagonal terms Q_{ii} are local biases and are not
        counted as couplers.

        Parameters
        ----------
        Q:
            QUBO dictionary.

        Returns
        -------
        int
            Number of non-zero off-diagonal QUBO edges.
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
        Run the minimal cardinality-sweep algorithm.

        Parameters
        ----------
        problem:
            Hitting set problem instance.

        simulator:
            Backend used to sample each QUBO.

        Returns
        -------
        AlgorithmResultSet
            Result containing the sampled minimal hitting sets.
        """
        mhs_found: List[AlgorithmResult] = []
        algorithm_metadata = {}

        ordered_universe = sorted(problem.universe)

        var_to_element = {
            idx: element
            for idx, element in enumerate(ordered_universe)
        }

        num_problem_vars = len(ordered_universe)

        # Work on a local copy. Do not mutate problem.test_cases.
        test_cases = self.prepare_test_cases(problem.test_cases)

        # ------------------------------------------------------------
        # Empty instance.
        #
        # The empty set is the unique minimal hitting set.
        # ------------------------------------------------------------
        if len(test_cases) == 0:
            return MinimalCardinalitySweepResultSet(
                problem=problem,
                mode=self.mode,
                num_qubits=0,
                num_couplers=0,
                solutions=[
                    AlgorithmResult(
                        set=frozenset(),
                        num_occurrences=1,
                    )
                ],
                metadata={
                    "preprocessing": {
                        "empty_instance": True,
                        "num_test_cases_original": len(problem.test_cases),
                        "num_test_cases_used": 0,
                        "start_cardinality_used": 0,
                    },
                    0: {
                        "direct_solution": True,
                        "simulator_metadata": {
                            "num_reads": 1,
                        },
                    },
                },
            )

        # ------------------------------------------------------------
        # Compute lower bound for the starting cardinality.
        # ------------------------------------------------------------
        start_cardinality, preprocessing_metadata = (
            self.compute_start_cardinality(
                universe=ordered_universe,
                test_cases=test_cases,
            )
        )

        algorithm_metadata["preprocessing"] = {
            **preprocessing_metadata,
            "prune_test_cases": self.config.prune_test_cases,
            "num_test_cases_original": len(problem.test_cases),
            "num_test_cases_used": len(test_cases),
        }

        # A minimal hitting set cannot have size larger than min(|U|, m).
        max_length = min(len(ordered_universe), len(test_cases))

        # ------------------------------------------------------------
        # Directly handle cardinality 1.
        # ------------------------------------------------------------
        len_1_solutions: Set[int] = set()

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
                    "num_qubits": 1,
                    "num_couplers": 0,
                    "simulator_metadata": {
                        "num_reads": 1,
                    },
                }

        # ------------------------------------------------------------
        # Prune length-1 solutions from test cases.
        # ------------------------------------------------------------
        pruned_length_one_results = prune_length_one_solutions(
            ordered_universe,
            test_cases,
            [[x] for x in len_1_solutions],
        )

        ordered_universe = pruned_length_one_results.pruned_universe
        test_cases = pruned_length_one_results.pruned_test_cases

        var_to_element = {
            idx: element
            for idx, element in enumerate(ordered_universe)
        }

        num_problem_vars = len(ordered_universe)

        # If the pruning step determines that there are no feasible solutions
        # after removing length-1 solutions, we can return early.
        if not pruned_length_one_results.feasible_without_singletons:
            return MinimalCardinalitySweepResultSet(
                problem=problem,
                mode=self.mode,
                num_qubits=1 if len(mhs_found) > 0 else 0,
                num_couplers=0,
                solutions=mhs_found,
                metadata=algorithm_metadata,
            )

        # ------------------------------------------------------------
        # Cardinality sweep.
        #
        # We do not stop after finding the first solutions, because this is
        # minimal enumeration, not minimum hitting-set search.
        # ------------------------------------------------------------
        max_num_qubits = 0
        max_num_couplers = 0

        first_qubo_cardinality = max(2, start_cardinality)

        for cardinality in range(first_qubo_cardinality, max_length + 1):
            iteration_metadata = {}
            mhs_counts: Dict[frozenset[int], int] = {}

            # --------------------------------------------------------
            # Construct QUBO.
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

            # --------------------------------------------------------
            # Count logical size of this QUBO.
            # --------------------------------------------------------
            num_qubits = self.count_num_qubits(
                Q=Q_matrix,
                num_problem_vars=num_problem_vars,
            )

            num_couplers = self.count_num_couplers(
                Q=Q_matrix,
            )

            max_num_qubits = max(max_num_qubits, num_qubits)
            max_num_couplers = max(max_num_couplers, num_couplers)

            iteration_metadata["num_qubits"] = num_qubits
            iteration_metadata["num_couplers"] = num_couplers

            # --------------------------------------------------------
            # Run simulator.
            # --------------------------------------------------------
            candidates = simulator.simulate(Q_matrix)

            iteration_metadata["simulator_metadata"] = candidates.metadata

            # --------------------------------------------------------
            # Keep only sampled ground states that are minimal hitting sets.
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

                if is_minimal_hitting_set(parsed_result, test_cases):
                    mhs_counts[parsed_result] = (
                        mhs_counts.get(parsed_result, 0)
                        + candidate.num_occurrences
                    )

            for mhs, count in mhs_counts.items():
                mhs_found.append(
                    AlgorithmResult(
                        set=mhs,
                        num_occurrences=count,
                    )
                )

            iteration_metadata["num_new_mhs_found"] = len(mhs_counts)

            algorithm_metadata[cardinality] = iteration_metadata

        # ------------------------------------------------------------
        # Store the maximum logical size across all generated QUBOs.
        # ------------------------------------------------------------
        if max_num_qubits == 0 and len(mhs_found) > 0:
            max_num_qubits = 1

        return MinimalCardinalitySweepResultSet(
            problem=problem,
            mode=self.mode,
            num_qubits=max_num_qubits,
            num_couplers=max_num_couplers,
            solutions=mhs_found,
            metadata=algorithm_metadata,
        )