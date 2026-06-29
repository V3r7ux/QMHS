import pytest

pytest.importorskip(
    "pysat",
    reason="python-sat is not installed; install it with `pip install python-sat`.",
)

from src.algorithms.classical.hitman_minimum import HitmanMinimumOPT
from src.core.problem import HittingSetProblem


def test_hitman_minimum_running_example():
    problem = HittingSetProblem(
        universe=[1, 2, 3],
        test_cases=[
            {1, 2},
            {1, 3},
        ],
        minimal_hitting_sets=[
            {1},
            {2, 3},
        ],
        name="running-example",
    )

    solutions, runtime = HitmanMinimumOPT().run(problem)

    assert set(map(frozenset, solutions)) == {
        frozenset({1}),
    }

    assert runtime >= 0.0


def test_hitman_minimum_multiple_optima():
    problem = HittingSetProblem(
        universe=[1, 2, 3, 4],
        test_cases=[
            {1, 2},
            {3, 4},
        ],
        minimal_hitting_sets=[
            {1, 3},
            {1, 4},
            {2, 3},
            {2, 4},
        ],
        name="multiple-optima",
    )

    solutions, _ = HitmanMinimumOPT().run(problem)

    assert set(map(frozenset, solutions)) == {
        frozenset({1, 3}),
        frozenset({1, 4}),
        frozenset({2, 3}),
        frozenset({2, 4}),
    }


def test_hitman_minimum_empty_family():
    problem = HittingSetProblem(
        universe=[1, 2, 3],
        test_cases=[],
        minimal_hitting_sets=[
            set(),
        ],
        name="empty-family",
    )

    solutions, _ = HitmanMinimumOPT().run(problem)

    assert solutions == [
        set(),
    ]