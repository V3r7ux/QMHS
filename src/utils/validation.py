import numpy as np

# Set helpers
def get_minimum_sets(collection):
    argmins = []
    minimum = np.inf
    for subset in collection:
        if len(subset) < minimum:
            minimum = len(subset)
            argmins = [subset]
        elif len(subset) == minimum:
            argmins.append(subset)
    return minimum, argmins

def is_hitting_set(solution, test_cases):
    misses = [solution.isdisjoint(s) for s in test_cases]
    return not any(misses)

def is_minimal_hitting_set(solution, test_cases):
    if not is_hitting_set(solution, test_cases):
        return False

    sol = set(solution)
    for x in solution:
        reduced = sol - {x}
        if all(len(reduced & tc) > 0 for tc in test_cases):
            return False
    return True