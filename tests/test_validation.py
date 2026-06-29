from src.utils.validation import (
    is_hitting_set,
    is_minimal_hitting_set,
    get_minimum_sets,
)


def test_running_example_hitting_sets():
    test_cases = [
        {1, 2},
        {1, 3},
    ]

    assert is_hitting_set({1}, test_cases)
    assert is_hitting_set({2, 3}, test_cases)
    assert is_hitting_set({1, 2, 3}, test_cases)

    assert not is_hitting_set({2}, test_cases)
    assert not is_hitting_set({3}, test_cases)
    assert not is_hitting_set(set(), test_cases)


def test_running_example_minimality():
    test_cases = [
        {1, 2},
        {1, 3},
    ]

    assert is_minimal_hitting_set({1}, test_cases)
    assert is_minimal_hitting_set({2, 3}, test_cases)

    assert not is_minimal_hitting_set({1, 2}, test_cases)
    assert not is_minimal_hitting_set({1, 3}, test_cases)
    assert not is_minimal_hitting_set({1, 2, 3}, test_cases)


def test_get_minimum_sets():
    collection = [
        {1, 2},
        {3},
        {4},
        {1, 2, 3},
    ]

    minimum, argmins = get_minimum_sets(collection)

    assert minimum == 1
    assert set(map(frozenset, argmins)) == {
        frozenset({3}),
        frozenset({4}),
    }