from itertools import product
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def qubo_energy(Q, active_variables):
    active = set(active_variables)
    energy = 0.0

    for (i, j), coeff in Q.items():
        if i in active and j in active:
            energy += coeff

    return energy


def num_qubo_variables(Q):
    if not Q:
        return 0

    return 1 + max(max(i, j) for i, j in Q)


def min_energy_for_problem_assignment(Q, selected_problem_vars, num_problem_vars):
    selected_problem_vars = set(selected_problem_vars)

    nvars = num_qubo_variables(Q)
    ancillas = list(range(num_problem_vars, nvars))

    best = float("inf")

    for bits in product([0, 1], repeat=len(ancillas)):
        active = set(selected_problem_vars)
        active.update(a for a, bit in zip(ancillas, bits) if bit)

        best = min(best, qubo_energy(Q, active))

    return best


def vars_for_candidate(candidate, universe):
    element_to_var = {
        element: idx
        for idx, element in enumerate(sorted(universe))
    }

    return {
        element_to_var[element]
        for element in candidate
    }