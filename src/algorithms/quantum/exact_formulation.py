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

from src.preprocessing.preprocessing_utils import remove_dominated_test_cases


@dataclass
class ExactFormulationConfig:
    """
    Configuration for the exact QUBO formulation of the minimal hitting set
    problem.

    Parameters
    ----------
    scale:
        Weight of the coverage constraints.

        For each test case C_j, the algorithm adds a penalty that is zero
        if and only if C_j is hit at least once.

    normalize:
        If True, divide all QUBO coefficients by the largest absolute
        coefficient.

        This can be useful for samplers that are sensitive to the absolute
        scale of the QUBO coefficients. The normalization constant is tracked
        internally so that the expected ground-state energy is rescaled
        consistently.

    prune_test_cases:
        If True, remove duplicate and dominated test cases before constructing
        the QUBO.

        If C_small is contained in C_big, then C_big is redundant, because
        every hitting set that hits C_small also hits C_big.
    """

    scale: float = 5.0
    normalize: bool = False
    prune_test_cases: bool = False


@dataclass
class ExactFormulationResultSet(AlgorithmResultSet):
    """
    Result set for the exact formulation.

    The num_qubits field stores the number of logical variables used by the
    QUBO, while num_couplers stores the number of non-zero off-diagonal QUBO
    terms, namely the logical QUBO edges.
    """
    
    def get_total_count(self, length):
        return self.metadata["simulator_metadata"]["num_reads"]


class ExactFormulationAlgorithm(BaseAlgorithm):
    """
    Exact QUBO formulation for sampling minimal hitting sets.

    This formulation does not fix the cardinality of the hitting set.

    For each test case C_j, it enforces:

        sum_{i in C_j} x_i >= 1

    by introducing binary ancilla variables and adding the penalty:

        (s_j - t_j - 1)^2

    where:

        s_j = sum_{i in C_j} x_i
        t_j = sum_k 2^k y_{j,k}

    If s_j >= 1, then the ancillas can choose t_j = s_j - 1 and the
    corresponding penalty is zero.

    Important
    ---------
    The ground-state manifold contains all hitting sets, not only minimal
    hitting sets. Therefore, after sampling, the algorithm filters the sampled
    ground states and keeps only those that are minimal hitting sets.
    """

    mode: str = "minimal"

    def __init__(self, config: ExactFormulationConfig | None = None):
        self.config = config or ExactFormulationConfig()

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

        Parameters
        ----------
        test_cases:
            Original test cases/conflicts from the problem.

        Returns
        -------
        List[Set[int]]
            Test cases used internally by the algorithm.

        Notes
        -----
        The original problem object is not modified.
        """
        prepared = [set(test_case) for test_case in test_cases]

        if self.config.prune_test_cases:
            prepared = remove_dominated_test_cases(prepared)

        return prepared

    # ------------------------------------------------------------------
    # QUBO construction
    # ------------------------------------------------------------------

    def construct_hamiltonian(
        self,
        universe: List[int],
        test_cases: List[Set[int]],
    ) -> Dict[Tuple[int, int], float]:
        """
        Construct the QUBO matrix for the exact minimal hitting set
        formulation.

        Parameters
        ----------
        universe:
            List of universe elements.

        test_cases:
            List of conflicts/test cases.

        Returns
        -------
        Dict[Tuple[int, int], float]
            QUBO matrix in dictionary form.

        QUBO structure
        --------------
        For each test case C_j, define:

            s_j = sum_{i in C_j} x_i
            t_j = sum_k 2^k y_{j,k}

        The penalty is:

            scale * (s_j - t_j - 1)^2

        This penalty is zero when C_j is hit at least once.

        Constant terms are omitted, as usual in QUBO dictionaries.
        """
        self.normalization_constant = 1.0

        Q: Dict[Tuple[int, int], float] = {}

        # ------------------------------------------------------------
        # Safety check.
        # ------------------------------------------------------------
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
        # Allocate ancillas for the coverage constraints.
        #
        # For a test case of size |C|, s_j ranges from 0 to |C|.
        # Since t_j = s_j - 1, t_j must represent values:
        #
        #     0, 1, ..., |C| - 1
        #
        # Therefore ceil(log2(|C|)) bits are enough.
        # ------------------------------------------------------------
        test_cases_offset: List[int] = []
        next_free_qubit = num_problem_vars

        for test_case in test_cases:
            test_cases_offset.append(next_free_qubit)

            num_ancilla = int(np.ceil(np.log2(len(test_case))))
            next_free_qubit += num_ancilla

        # ------------------------------------------------------------
        # Coverage constraints.
        # ------------------------------------------------------------
        for j, test_case in enumerate(test_cases):
            ancilla_offset = test_cases_offset[j]
            num_ancilla = int(np.ceil(np.log2(len(test_case))))

            # Convert real elements into compact qubit indices.
            tc_vars = sorted(element_to_var[element] for element in test_case)

            # --------------------------------------------------------
            # t_j^2, where:
            #
            #     t_j = sum_k 2^k y_{j,k}
            # --------------------------------------------------------
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

            # --------------------------------------------------------
            # + 2 t_j
            # --------------------------------------------------------
            for k in range(num_ancilla):
                yk = ancilla_offset + k

                Q[(yk, yk)] = Q.get((yk, yk), 0.0) + (
                    2.0 * (2.0 ** k) * self.config.scale
                )

            # --------------------------------------------------------
            # s_j^2, where:
            #
            #     s_j = sum_{i in C_j} x_i
            # --------------------------------------------------------
            for idx, i in enumerate(tc_vars):
                Q[(i, i)] = Q.get((i, i), 0.0) + self.config.scale

                for i_prime in tc_vars[idx + 1:]:
                    Q[(i, i_prime)] = Q.get((i, i_prime), 0.0) + (
                        2.0 * self.config.scale
                    )

            # --------------------------------------------------------
            # - 2 s_j
            # --------------------------------------------------------
            for i in tc_vars:
                Q[(i, i)] = Q.get((i, i), 0.0) - (
                    2.0 * self.config.scale
                )

            # --------------------------------------------------------
            # - 2 s_j t_j
            # --------------------------------------------------------
            for i in tc_vars:
                for k in range(num_ancilla):
                    yk = ancilla_offset + k

                    Q[(i, yk)] = Q.get((i, yk), 0.0) - (
                        2.0 * (2.0 ** k) * self.config.scale
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
    ) -> float:
        """
        Compute the expected ground-state energy.

        Parameters
        ----------
        num_test_cases:
            Number of test cases used in the QUBO.

        Returns
        -------
        float
            Expected ground-state energy after optional normalization.

        Notes
        -----
        Since the constant terms are omitted from the QUBO, every satisfied
        test case contributes:

            - scale

        to the variable-dependent energy.

        Therefore, every hitting set has energy:

            - num_test_cases * scale
        """
        energy = -num_test_cases * self.config.scale

        return energy / self.normalization_constant

    def parse_candidate_sample(
        self,
        sample: Set[int],
        var_to_element: Dict[int, int],
        num_problem_vars: int,
    ) -> frozenset[int]:
        """
        Convert a sampled qubit configuration into a set of original universe
        elements.

        Parameters
        ----------
        sample:
            Set of active qubit indices returned by the simulator.

        var_to_element:
            Mapping from compact qubit indices to original universe elements.

        num_problem_vars:
            Number of real problem variables. Qubits with index greater than
            or equal to this value are ancillas and are ignored.

        Returns
        -------
        frozenset[int]
            Candidate hitting set expressed using original universe labels.
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
        Count the total number of qubits appearing in the QUBO.

        Parameters
        ----------
        Q:
            QUBO dictionary.

        num_problem_vars:
            Number of original problem variables.

        Returns
        -------
        int
            Number of qubits used by the formulation.
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
        Run the exact QUBO formulation.

        Parameters
        ----------
        problem:
            Hitting set problem instance.

        simulator:
            Backend used to sample the QUBO.

        Returns
        -------
        AlgorithmResultSet
            Result containing the sampled minimal hitting sets and metadata.
        """
        algorithm_metadata = {}

        ordered_universe = sorted(problem.universe)

        var_to_element = {
            idx: element
            for idx, element in enumerate(ordered_universe)
        }

        num_problem_vars = len(ordered_universe)

        # Work on a local copy of the test cases.
        # Do not mutate problem.test_cases.
        test_cases = self.prepare_test_cases(problem.test_cases)

        algorithm_metadata["preprocessing"] = {
            "prune_test_cases": self.config.prune_test_cases,
            "num_test_cases_original": len(problem.test_cases),
            "num_test_cases_used": len(test_cases),
        }

        # ------------------------------------------------------------
        # Empty instance.
        #
        # If there are no test cases, the empty set is the unique minimal
        # hitting set.
        # ------------------------------------------------------------
        if len(test_cases) == 0:
            return ExactFormulationResultSet(
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
                    **algorithm_metadata,
                    "num_couplers": 0,
                    "simulator_metadata": {
                        "num_reads": 1,
                    },
                },
            )

        # ------------------------------------------------------------
        # Construct QUBO.
        # ------------------------------------------------------------
        start_matrix = time.time()

        Q_matrix = self.construct_hamiltonian(
            universe=ordered_universe,
            test_cases=test_cases,
        )

        end_matrix = time.time()

        algorithm_metadata["matrix_construction_time"] = (
            end_matrix - start_matrix
        )

        # ------------------------------------------------------------
        # Count logical size.
        # ------------------------------------------------------------
        num_qubits = self.count_num_qubits(
            Q=Q_matrix,
            num_problem_vars=num_problem_vars,
        )

        num_couplers = self.count_num_couplers(
            Q=Q_matrix,
        )

        algorithm_metadata["num_qubits"] = num_qubits
        algorithm_metadata["num_couplers"] = num_couplers

        # ------------------------------------------------------------
        # Run simulator.
        # ------------------------------------------------------------
        candidates = simulator.simulate(Q_matrix)

        algorithm_metadata["simulator_metadata"] = candidates.metadata

        # ------------------------------------------------------------
        # Filter sampled ground states and keep only minimal hitting sets.
        # ------------------------------------------------------------
        mhs_counts: Dict[frozenset[int], int] = {}

        gs_energy = self.ground_state_energy(
            num_test_cases=len(test_cases),
        )

        tol = 1.0e-8

        for candidate in candidates.results:
            # If the sampled state is not in the ground-state manifold,
            # skip it.
            if candidate.energy > gs_energy + tol:
                continue

            # Keep only the real problem variables.
            parsed_result = self.parse_candidate_sample(
                sample=candidate.sample,
                var_to_element=var_to_element,
                num_problem_vars=num_problem_vars,
            )

            # The exact formulation has all hitting sets in the ground state.
            # We explicitly filter only the minimal ones.
            if is_minimal_hitting_set(parsed_result, test_cases):
                mhs_counts[parsed_result] = (
                    mhs_counts.get(parsed_result, 0)
                    + candidate.num_occurrences
                )

        mhs_found = [
            AlgorithmResult(
                set=mhs,
                num_occurrences=count,
            )
            for mhs, count in mhs_counts.items()
        ]

        return ExactFormulationResultSet(
            problem=problem,
            mode=self.mode,
            num_qubits=num_qubits,
            num_couplers=num_couplers,
            solutions=mhs_found,
            metadata=algorithm_metadata,
        )