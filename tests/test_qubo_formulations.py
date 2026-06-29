import pytest

from src.algorithms.quantum.exact_formulation import (
    ExactFormulationAlgorithm,
    ExactFormulationConfig,
)

from src.algorithms.quantum.cardinality_sweep import (
    MinimalCardinalitySweepAlgorithm,
    MinimalCardinalitySweepConfig,
)

from src.algorithms.quantum.minimum_cardinality_sweep import (
    MinimumPUBOAlgorithm,
    MinimumPUBOConfig,
)

from src.algorithms.quantum.recursive_pubo import (
    RecursivePUBOAlgorithm,
    RecursivePUBOConfig,
)

from src.core.result import AlgorithmResult

from conftest import (
    min_energy_for_problem_assignment,
    vars_for_candidate,
)


def test_exact_formulation_ground_states_are_hitting_sets_running_example():
    universe = [1, 2, 3]
    test_cases = [
        {1, 2},
        {1, 3},
    ]

    algorithm = ExactFormulationAlgorithm(
        ExactFormulationConfig(
            scale=1.0,
            normalize=False,
        )
    )

    Q = algorithm.construct_hamiltonian(
        universe=universe,
        test_cases=test_cases,
    )

    ground = algorithm.ground_state_energy(
        num_test_cases=len(test_cases),
    )

    candidates = [
        set(),
        {1},
        {2},
        {3},
        {1, 2},
        {1, 3},
        {2, 3},
        {1, 2, 3},
    ]

    hitting_sets = [
        {1},
        {1, 2},
        {1, 3},
        {2, 3},
        {1, 2, 3},
    ]

    for candidate in candidates:
        energy = min_energy_for_problem_assignment(
            Q=Q,
            selected_problem_vars=vars_for_candidate(candidate, universe),
            num_problem_vars=len(universe),
        )

        if candidate in hitting_sets:
            assert energy == pytest.approx(ground)
        else:
            assert energy > ground


def test_cardinality_sweep_k1_and_k2_running_example():
    universe = [1, 2, 3]
    test_cases = [
        {1, 2},
        {1, 3},
    ]

    algorithm = MinimalCardinalitySweepAlgorithm(
        MinimalCardinalitySweepConfig(
            scale=1.0,
            cardinality_scale=1.0,
            normalize=False,
        )
    )

    Q_k1 = algorithm.construct_hamiltonian(
        universe=universe,
        test_cases=test_cases,
        cardinality=1,
    )

    ground_k1 = algorithm.ground_state_energy(
        num_test_cases=len(test_cases),
        cardinality=1,
    )

    assert min_energy_for_problem_assignment(
        Q=Q_k1,
        selected_problem_vars=vars_for_candidate({1}, universe),
        num_problem_vars=len(universe),
    ) == pytest.approx(ground_k1)

    assert min_energy_for_problem_assignment(
        Q=Q_k1,
        selected_problem_vars=vars_for_candidate({2, 3}, universe),
        num_problem_vars=len(universe),
    ) > ground_k1

    Q_k2 = algorithm.construct_hamiltonian(
        universe=universe,
        test_cases=test_cases,
        cardinality=2,
    )

    ground_k2 = algorithm.ground_state_energy(
        num_test_cases=len(test_cases),
        cardinality=2,
    )

    for candidate in [
        {1, 2},
        {1, 3},
        {2, 3},
    ]:
        assert min_energy_for_problem_assignment(
            Q=Q_k2,
            selected_problem_vars=vars_for_candidate(candidate, universe),
            num_problem_vars=len(universe),
        ) == pytest.approx(ground_k2)

    assert min_energy_for_problem_assignment(
        Q=Q_k2,
        selected_problem_vars=vars_for_candidate({1}, universe),
        num_problem_vars=len(universe),
    ) > ground_k2


def test_minimum_formulation_adaptive_ancillas_reduce_logical_size():
    universe = list(range(1, 17))
    test_cases = [
        set(universe),
    ]

    standard = MinimumPUBOAlgorithm(
        MinimumPUBOConfig(
            use_cardinality_limited_coverage_ancillas=False,
            normalize=False,
        )
    )

    adaptive = MinimumPUBOAlgorithm(
        MinimumPUBOConfig(
            use_cardinality_limited_coverage_ancillas=True,
            normalize=False,
        )
    )

    Q_standard = standard.construct_hamiltonian(
        universe=universe,
        test_cases=test_cases,
        cardinality=2,
    )

    Q_adaptive = adaptive.construct_hamiltonian(
        universe=universe,
        test_cases=test_cases,
        cardinality=2,
    )

    n_standard = 1 + max(max(i, j) for i, j in Q_standard)
    n_adaptive = 1 + max(max(i, j) for i, j in Q_adaptive)

    assert n_standard == len(universe) + 4
    assert n_adaptive == len(universe) + 1
    assert n_adaptive < n_standard


def test_recursive_pubo_penalizes_supersets_of_previous_mhs():
    universe = [1, 2, 3]
    test_cases = [
        {1, 2},
        {1, 3},
    ]

    algorithm = RecursivePUBOAlgorithm(
        RecursivePUBOConfig(
            scale=1.0,
            cardinality_scale=1.0,
            reuse_element_recursive=1.0,
            constraint_strength=3.0,
            normalize=False,
        )
    )

    previous = [
        AlgorithmResult(
            set=frozenset({1}),
        )
    ]

    Q = algorithm.construct_hamiltonian(
        universe=universe,
        test_cases=test_cases,
        cardinality=2,
        mhs_found=previous,
    )

    ground = algorithm.ground_state_energy(
        num_test_cases=len(test_cases),
        cardinality=2,
    )

    assert min_energy_for_problem_assignment(
        Q=Q,
        selected_problem_vars=vars_for_candidate({2, 3}, universe),
        num_problem_vars=len(universe),
    ) == pytest.approx(ground)

    assert min_energy_for_problem_assignment(
        Q=Q,
        selected_problem_vars=vars_for_candidate({1, 2}, universe),
        num_problem_vars=len(universe),
    ) == pytest.approx(ground + 1.0)

    assert min_energy_for_problem_assignment(
        Q=Q,
        selected_problem_vars=vars_for_candidate({1, 3}, universe),
        num_problem_vars=len(universe),
    ) == pytest.approx(ground + 1.0)