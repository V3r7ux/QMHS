import pytest

from src.preprocessing.preprocessing_utils import (
    remove_duplicate_test_cases,
    remove_dominated_test_cases,
    validate_instance,
    compute_lower_bounds,
    greedy_upper_bound,
    prune_length_one_solutions,
)


def test_remove_duplicate_test_cases():
    test_cases = [
        {1, 2},
        {2, 1},
        {3},
    ]

    assert remove_duplicate_test_cases(test_cases) == [
        {3},
        {1, 2},
    ]


def test_remove_dominated_test_cases():
    test_cases = [
        {1, 2},
        {1, 2, 3},
        {4, 5},
        {4, 5, 6},
    ]

    reduced = remove_dominated_test_cases(test_cases)

    assert set(map(frozenset, reduced)) == {
        frozenset({1, 2}),
        frozenset({4, 5}),
    }


def test_validate_instance_rejects_empty_conflict():
    with pytest.raises(ValueError, match="empty test case"):
        validate_instance(
            universe=[1, 2, 3],
            test_cases=[{1}, set()],
        )


def test_validate_instance_rejects_elements_outside_universe():
    with pytest.raises(ValueError, match="outside the universe"):
        validate_instance(
            universe=[1, 2, 3],
            test_cases=[{1, 4}],
        )


def test_lower_bounds_running_example():
    universe = [1, 2, 3]
    test_cases = [
        {1, 2},
        {1, 3},
    ]

    bounds = compute_lower_bounds(
        universe=universe,
        test_cases=test_cases,
        use_lp=True,
    )

    assert bounds.trivial == 1
    assert bounds.max_degree == 1
    assert bounds.sum_degree == 1
    assert bounds.greedy_packing == 1
    assert bounds.lp_relaxation == 1
    assert bounds.value == 1


def test_greedy_upper_bound_returns_valid_hitting_set():
    universe = [1, 2, 3, 4]
    test_cases = [
        {1, 2},
        {2, 3},
        {3, 4},
    ]

    solution = greedy_upper_bound(universe, test_cases)

    assert all(solution & test_case for test_case in test_cases)


def test_prune_length_one_solutions_running_example():
    universe = [1, 2, 3]
    test_cases = [
        {1, 2},
        {1, 3},
    ]

    result = prune_length_one_solutions(
        universe=universe,
        test_cases=test_cases,
        singleton_solutions=[{1}],
    )

    assert result.pruned_universe == [2, 3]
    assert result.pruned_test_cases == [
        {2},
        {3},
    ]
    assert result.removed_elements == {1}
    assert result.feasible_without_singletons