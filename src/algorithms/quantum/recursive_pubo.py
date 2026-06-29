from __future__ import annotations

from dataclasses import dataclass, field
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
class AncillaProductManager:
    """
    Helper class used to transform product terms into quadratic terms.

    The recursive PUBO penalty contains products of the form:

        prod_{i in M} x_i

    where M is a previously found minimal hitting set.

    Since the simulator expects a QUBO, products of more than two variables
    are reduced using ancilla variables.

    Parameters
    ----------
    next_free_qubit:
        First available qubit index after the problem variables and the
        coverage ancillas.

    constraint_strength:
        Penalty strength used to enforce the equality:

            z = a b

        where z is the ancilla variable representing the product of a and b.

    product_cache:
        Dictionary used to avoid creating multiple ancillas for the same
        product. If the product x_a x_b has already been encoded, the same
        ancilla is reused.
    """

    next_free_qubit: int
    constraint_strength: float
    product_cache: Dict[Tuple[int, int], int] = field(default_factory=dict)

    def get_product_ancilla(
        self,
        Q: Dict[Tuple[int, int], float],
        a: int,
        b: int,
    ) -> int:
        """
        Return an ancilla representing the product x_a x_b.

        If the product has already been encoded, reuse the previous ancilla.
        Otherwise, create a new ancilla z and add the standard QUBO penalty
        enforcing:

            z = x_a x_b

        Parameters
        ----------
        Q:
            QUBO dictionary updated in-place.

        a:
            First variable index.

        b:
            Second variable index.

        Returns
        -------
        int
            Qubit index of the ancilla representing x_a x_b.
        """
        ordered_pair = (min(a, b), max(a, b))

        if ordered_pair in self.product_cache:
            return self.product_cache[ordered_pair]

        z = self.next_free_qubit
        self.product_cache[ordered_pair] = z
        self.next_free_qubit += 1

        strength = self.constraint_strength

        # Penalty for z = a b:
        #
        #     strength * (3z + ab - 2az - 2bz)
        #
        # This is zero only when z = a b.
        Q[(z, z)] = Q.get((z, z), 0.0) + 3.0 * strength
        Q[(ordered_pair[0], z)] = Q.get((ordered_pair[0], z), 0.0) - 2.0 * strength
        Q[(ordered_pair[1], z)] = Q.get((ordered_pair[1], z), 0.0) - 2.0 * strength
        Q[ordered_pair] = Q.get(ordered_pair, 0.0) + strength

        return z


@dataclass
class RecursivePUBOConfig:
    """
    Configuration for the recursive PUBO/QUBO algorithm that enumerates
    minimal hitting sets.

    Parameters
    ----------
    scale:
        Weight of the coverage constraints.

    cardinality_scale:
        Weight of the cardinality penalty:

            (sum_i x_i - K)^2

        where K is the target cardinality of the current sweep iteration.

    reuse_element_recursive:
        Weight of the recursive penalty used to remove non-minimal supersets
        of already found minimal hitting sets.

    constraint_strength:
        Penalty strength used when reducing higher-order product terms to QUBO
        form with ancilla variables.

    normalize:
        If True, divide all QUBO coefficients by the largest absolute
        coefficient.

    prune_test_cases:
        If True, remove duplicate and dominated test cases before constructing
        the QUBO.

    use_lower_bound:
        If True, start the cardinality sweep from a classical lower bound
        instead of starting from K = 1.

    use_lp_lower_bound:
        If True, include the LP-relaxation lower bound.

    use_cardinality_limited_coverage_ancillas:
        If True, reduce the number of coverage ancillas using the current
        cardinality K.

        Since the cardinality constraint enforces:

            sum_i x_i = K

        the overlap between a candidate solution and a conflict C_j is bounded by:

            |C_j cap X| <= min(|C_j|, K)

        Therefore the auxiliary integer t_j only needs to encode values up to:

            min(|C_j|, K) - 1

        instead of:

            |C_j| - 1

        This reduces the number of coverage ancillas from:

            ceil(log2(|C_j|))

        to:

            ceil(log2(min(|C_j|, K)))

        for each conflict.
    """

    scale: float = 1.0
    cardinality_scale: float = 1.0
    reuse_element_recursive: float = 1.0
    constraint_strength: float = 3.0
    normalize: bool = False
    prune_test_cases: bool = False

    use_lower_bound: bool = True
    use_lp_lower_bound: bool = False

    use_cardinality_limited_coverage_ancillas: bool = False


@dataclass
class RecursivePUBOResultSet(AlgorithmResultSet):
    """
    Result set for the recursive minimal hitting set PUBO algorithm.

    The benchmark code can ask how many reads were used at a given
    cardinality. This method extracts that information from the metadata.

    The num_qubits and num_couplers fields store the maximum logical size
    reached across all QUBOs generated during the cardinality sweep.
    """

    def get_total_count(self, length):
        return self.metadata[length]["simulator_metadata"]["num_reads"]


class RecursivePUBOAlgorithm(BaseAlgorithm):
    """
    Recursive PUBO/QUBO algorithm for enumerating minimal hitting sets.

    The algorithm sweeps over cardinalities:

        K = lower_bound, lower_bound + 1, ..., max_length

    For each cardinality K, it constructs a QUBO whose ground states should
    correspond to hitting sets of size K.

    To enforce minimality recursively, every minimal hitting set found at
    smaller cardinality is penalized through the term:

        prod_{i in M} x_i

    Therefore, any future candidate that contains an already found MHS M gets
    positive energy and is removed from the ground-state manifold.

    This class is for enumerating minimal hitting sets, not only for finding
    the minimum hitting set.
    """

    mode: str = "minimal"

    def __init__(self, config: RecursivePUBOConfig | None = None):
        self.config = config or RecursivePUBOConfig()

        # Used only if normalize=True.
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

        The original problem object is not modified.

        Parameters
        ----------
        test_cases:
            Original test cases/conflicts.

        Returns
        -------
        List[Set[int]]
            Test cases used internally by the algorithm.
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

        For minimal enumeration, a lower bound is useful only to skip
        impossible cardinalities below the true minimum hitting set size.

        Unlike the minimum algorithm, we do not use a greedy upper bound to
        stop the sweep, because minimal hitting sets can be larger than the
        minimum hitting set.

        Parameters
        ----------
        universe:
            Sorted list of universe elements.

        test_cases:
            Test cases after optional pruning.

        Returns
        -------
        tuple[int, dict]
            start_cardinality:
                First cardinality to test.

            preprocessing_metadata:
                Dictionary containing lower-bound information.
        """
        if len(test_cases) == 0:
            metadata = {
                "empty_instance": True,
                "start_cardinality": 0,
            }
            return 0, metadata

        start_preprocessing = time.time()

        preprocessing = preprocess_hitting_set_instance(
            universe=universe,
            test_cases=test_cases,
            use_lp_lower_bound=self.config.use_lp_lower_bound,
            prune_dominated=False,
        )

        end_preprocessing = time.time()

        preprocessing_metadata = preprocessing.as_dict()
        preprocessing_metadata["empty_instance"] = False
        preprocessing_metadata["use_lower_bound"] = self.config.use_lower_bound
        preprocessing_metadata["use_lp_lower_bound"] = self.config.use_lp_lower_bound
        preprocessing_metadata["preprocessing_time"] = (
            end_preprocessing - start_preprocessing
        )
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

        An element forms a size-1 hitting set if it belongs to every test case.

        Parameters
        ----------
        test_cases:
            Test cases/conflicts.

        Returns
        -------
        Set[int]
            Elements that alone hit all test cases.
        """
        if len(test_cases) == 0:
            return set()

        return set.intersection(*test_cases)

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

        In the standard formulation, for a conflict C_j, the auxiliary integer
        t_j represents:

            t_j = |C_j cap X| - 1

        Therefore t_j must encode values from:

            0 to |C_j| - 1

        If cardinality-limited coverage ancillas are enabled, then during the
        cardinality sweep we also know that:

            |X| = K

        so:

            |C_j cap X| <= min(|C_j|, K)

        Therefore t_j only needs to encode values from:

            0 to min(|C_j|, K) - 1

        Parameters
        ----------
        test_case_size:
            Size of the conflict/test case.

        cardinality:
            Current target cardinality K.

        Returns
        -------
        int
            Number of binary ancilla variables needed to encode t_j.
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
        mhs_found: List[AlgorithmResult],
    ) -> Dict[Tuple[int, int], float]:
        """
        Construct the recursive PUBO/QUBO Hamiltonian.

        Parameters
        ----------
        universe:
            List of universe elements.

        test_cases:
            List of conflicts/test cases.

        cardinality:
            Target cardinality K.

        mhs_found:
            Minimal hitting sets already found at previous cardinalities.

        Returns
        -------
        Dict[Tuple[int, int], float]
            QUBO matrix in dictionary form.

        Hamiltonian structure
        ---------------------
        The QUBO contains:

        1. Coverage constraints:

               (s_j - t_j - 1)^2

           where:

               s_j = sum_{i in C_j} x_i
               t_j = binary-encoded auxiliary integer

           This makes it possible to have zero coverage penalty whenever
           the test case C_j is hit at least once.

        2. Cardinality penalty:

               (sum_i x_i - K)^2

        3. Recursive minimality penalty:

               sum_{M already found} prod_{i in M} x_i

           This penalizes any candidate that fully contains a previously
           found minimal hitting set M.
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
        #
        # Standard behavior:
        #     num_ancilla = ceil(log2(|C_j|))
        #
        # Cardinality-limited behavior:
        #     num_ancilla = ceil(log2(min(|C_j|, K)))
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
        # Recursive minimality penalty.
        #
        # Defensive note:
        #     Only MHS fully contained in the current universe can be encoded.
        #     This matters after length-1 pruning, because singleton MHS are
        #     kept in the final result list but their elements may have been
        #     removed from the reduced universe.
        # ------------------------------------------------------------
        product_manager = AncillaProductManager(
            next_free_qubit=next_free_qubit,
            constraint_strength=self.config.constraint_strength,
        )

        for mhs in mhs_found:
            if not mhs.set.issubset(universe_set):
                continue

            mhs_vars = sorted(element_to_var[element] for element in mhs.set)
            len_mhs = len(mhs_vars)

            if len_mhs == 0:
                continue

            if len_mhs == 1:
                i = mhs_vars[0]

                Q[(i, i)] = Q.get((i, i), 0.0) + (
                    self.config.reuse_element_recursive
                )

            elif len_mhs == 2:
                i, j = mhs_vars

                Q[(i, j)] = Q.get((i, j), 0.0) + (
                    self.config.reuse_element_recursive
                )

            elif len_mhs > 2 and self.config.constraint_strength > 0:
                z = mhs_vars[0]

                for i in mhs_vars[1:-1]:
                    z = product_manager.get_product_ancilla(Q, z, i)

                last_var = mhs_vars[-1]

                ordered_pair = (min(last_var, z), max(last_var, z))

                Q[ordered_pair] = Q.get(ordered_pair, 0.0) + (
                    self.config.reuse_element_recursive
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
        Compute the expected ground-state energy for a valid minimal hitting
        set at the current cardinality.

        The recursive penalty is zero for candidates that do not contain
        previously found minimal hitting sets, so the expected ground-state
        energy is the same as in the non-recursive formulation.

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
        Convert a sampled qubit configuration into a set of original elements.

        Ancilla qubits are ignored.

        Parameters
        ----------
        sample:
            Set of active qubit indices returned by the simulator.

        var_to_element:
            Mapping from compact qubit indices to original universe elements.

        num_problem_vars:
            Number of real problem variables.

        Returns
        -------
        frozenset[int]
            Candidate hitting set expressed in original universe labels.
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
        Run the recursive minimal hitting set PUBO algorithm.

        Parameters
        ----------
        problem:
            Hitting set problem instance.

        simulator:
            Backend used to sample the QUBO.

        Returns
        -------
        AlgorithmResultSet
            Result containing the minimal hitting sets sampled by the
            algorithm and metadata for each cardinality.
        """
        mhs_found: List[AlgorithmResult] = []

        # This list contains only the MHS that can be encoded as recursive
        # penalties in the current reduced problem.
        #
        # Length-1 solutions found before prune_length_one_solutions are kept
        # in mhs_found for the final output, but they are not added here
        # because their elements may be removed from the reduced universe.
        mhs_for_recursive_penalty: List[AlgorithmResult] = []

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
            mhs_found.append(AlgorithmResult(frozenset(), 1))

            algorithm_metadata["preprocessing"] = {
                "empty_instance": True,
                "start_cardinality_used": 0,
                "num_test_cases_original": len(problem.test_cases),
                "num_test_cases_used": 0,
            }

            return RecursivePUBOResultSet(
                problem=problem,
                mode=self.mode,
                num_qubits=0,
                num_couplers=0,
                solutions=mhs_found,
                metadata=algorithm_metadata,
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
            "num_test_cases_original": len(problem.test_cases),
            "num_test_cases_used": len(test_cases),
        }

        # A minimal hitting set cannot have size larger than min(|U|, m),
        # because one can always build a hitting set by selecting at most one
        # element from each test case, and any minimal subset of it is no larger.
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
            return RecursivePUBOResultSet(
                problem=problem,
                mode=self.mode,
                num_qubits=1 if len(mhs_found) > 0 else 0,
                num_couplers=0,
                solutions=mhs_found,
                metadata=algorithm_metadata,
            )

        # ------------------------------------------------------------
        # Recursive cardinality sweep.
        #
        # Unlike the minimum class, we do not stop at the first successful
        # cardinality. We continue because there may be larger minimal hitting
        # sets.
        # ------------------------------------------------------------
        max_num_qubits = 0
        max_num_couplers = 0

        first_qubo_cardinality = max(2, start_cardinality)

        for cardinality in range(first_qubo_cardinality, max_length + 1):
            iteration_metadata = {}
            mhs_counts: Dict[frozenset[int], int] = {}

            # --------------------------------------------------------
            # Build QUBO with recursive penalties from previous MHS.
            #
            # Only MHS found inside the reduced problem are used as recursive
            # penalties. Direct length-1 MHS are already handled by the
            # pruning step and may not belong to the reduced universe anymore.
            # --------------------------------------------------------
            start_matrix = time.time()

            Q_matrix = self.construct_hamiltonian(
                universe=ordered_universe,
                test_cases=test_cases,
                cardinality=cardinality,
                mhs_found=mhs_for_recursive_penalty,
            )

            end_matrix = time.time()

            iteration_metadata["matrix_construction_time"] = (
                end_matrix - start_matrix
            )

            iteration_metadata["num_previous_mhs_penalized"] = (
                len(mhs_for_recursive_penalty)
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
            # Filter sampled ground states.
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

            # Add newly found MHS to the global list and to the recursive
            # penalty list. These MHS were found after length-1 pruning, so
            # they are expressed in the reduced universe and can safely be
            # used in later recursive QUBOs.
            for mhs, count in mhs_counts.items():
                result = AlgorithmResult(
                    set=mhs,
                    num_occurrences=count,
                )

                mhs_found.append(result)
                mhs_for_recursive_penalty.append(result)

            iteration_metadata["num_new_mhs_found"] = len(mhs_counts)

            algorithm_metadata[cardinality] = iteration_metadata

        # ------------------------------------------------------------
        # Store the maximum logical size across all generated QUBOs.
        # ------------------------------------------------------------
        if max_num_qubits == 0 and len(mhs_found) > 0:
            max_num_qubits = 1

        return RecursivePUBOResultSet(
            problem=problem,
            mode=self.mode,
            num_qubits=max_num_qubits,
            num_couplers=max_num_couplers,
            solutions=mhs_found,
            metadata=algorithm_metadata,
        )