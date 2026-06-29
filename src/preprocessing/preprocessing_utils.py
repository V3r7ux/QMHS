from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Dict, Iterable, List, Set

import numpy as np


@dataclass(frozen=True)
class LowerBoundResult:
    """
    Container for all lower bounds computed for a hitting set instance.

    Parameters
    ----------
    trivial:
        Trivial lower bound. It is 0 if there are no test cases,
        otherwise it is 1.

    max_degree:
        Lower bound obtained from the maximum number of test cases that
        can be hit by a single universe element.

    sum_degree:
        Lower bound obtained by sorting element degrees in decreasing order
        and asking how many best possible elements would be needed to cover
        the number of test cases.

    greedy_packing:
        Lower bound obtained by greedily constructing a set of pairwise
        disjoint test cases. Since one element can hit at most one test case
        in a disjoint packing, the size of the packing is a lower bound.

    lp_relaxation:
        Optional lower bound obtained by solving the LP relaxation of the
        minimum hitting set problem. If not computed, it is None.
    """

    trivial: int
    max_degree: int
    sum_degree: int
    greedy_packing: int
    lp_relaxation: int | None = None

    @property
    def value(self) -> int:
        """
        Return the strongest available lower bound.

        Returns
        -------
        int
            The maximum among all computed lower bounds.
        """
        values = [
            self.trivial,
            self.max_degree,
            self.sum_degree,
            self.greedy_packing,
        ]

        if self.lp_relaxation is not None:
            values.append(self.lp_relaxation)

        return max(values)

    def as_dict(self) -> Dict[str, int | None]:
        """
        Convert the lower-bound result to a dictionary.

        This is useful for saving the preprocessing information inside
        algorithm metadata.

        Returns
        -------
        Dict[str, int | None]
            Dictionary containing all lower bounds and the final selected one.
        """
        return {
            "trivial": self.trivial,
            "max_degree": self.max_degree,
            "sum_degree": self.sum_degree,
            "greedy_packing": self.greedy_packing,
            "lp_relaxation": self.lp_relaxation,
            "value": self.value,
        }


def normalize_test_cases(
    test_cases: Iterable[Iterable[int]],
) -> List[Set[int]]:
    """
    Convert test cases to a clean list of sets.

    Empty test cases are not removed here silently, because an empty test case
    means that the instance is infeasible: no hitting set can hit an empty set.

    Parameters
    ----------
    test_cases:
        Iterable containing the test cases / conflicts.
        Each test case is itself an iterable of universe elements.

    Returns
    -------
    List[Set[int]]
        A list where every test case is converted to a Python set.

    Raises
    ------
    ValueError
        If one of the test cases is empty.
    """
    normalized: List[Set[int]] = []

    for test_case in test_cases:
        current = set(test_case)

        if len(current) == 0:
            raise ValueError(
                "The instance contains an empty test case. "
                "No hitting set can hit an empty set, so the instance is infeasible."
            )

        normalized.append(current)

    return normalized


def remove_duplicate_test_cases(
    test_cases: Iterable[Iterable[int]],
) -> List[Set[int]]:
    """
    Remove duplicate test cases.

    Parameters
    ----------
    test_cases:
        Iterable containing the test cases / conflicts.

    Returns
    -------
    List[Set[int]]
        Test cases with duplicates removed.
    """
    unique = list({frozenset(test_case) for test_case in test_cases})
    unique.sort(key=lambda test_case: (len(test_case), sorted(test_case)))

    return [set(test_case) for test_case in unique]


def remove_dominated_test_cases(
    test_cases: Iterable[Iterable[int]],
) -> List[Set[int]]:
    """
    Remove dominated test cases.

    A test case C_big is dominated by C_small if:

        C_small subseteq C_big

    In this case, C_big can be removed because every hitting set that hits
    C_small automatically hits C_big.

    Example
    -------
    If we have:

        {1, 2}
        {1, 2, 3}

    then {1, 2, 3} is redundant and can be removed.

    Parameters
    ----------
    test_cases:
        Iterable containing the test cases / conflicts.

    Returns
    -------
    List[Set[int]]
        Test cases after duplicate and dominated test cases are removed.
    """
    normalized = remove_duplicate_test_cases(test_cases)

    # Smaller sets must be considered first, because they can dominate
    # larger sets.
    normalized.sort(key=lambda test_case: (len(test_case), sorted(test_case)))

    minimal_cases: List[Set[int]] = []

    for candidate in normalized:
        is_dominated = any(
            existing.issubset(candidate)
            for existing in minimal_cases
        )

        if not is_dominated:
            minimal_cases.append(candidate)

    return minimal_cases


def validate_instance(
    universe: Iterable[int],
    test_cases: Iterable[Iterable[int]],
) -> tuple[List[int], List[Set[int]]]:
    """
    Validate and normalize a hitting set instance.

    This function checks that every element appearing in the test cases
    also belongs to the universe.

    Parameters
    ----------
    universe:
        Iterable containing all available universe elements.

    test_cases:
        Iterable containing the test cases / conflicts.

    Returns
    -------
    tuple[List[int], List[Set[int]]]
        The sorted universe and the normalized test cases.

    Raises
    ------
    ValueError
        If a test case contains an element outside the universe.
    """
    ordered_universe = sorted(set(universe))
    universe_set = set(ordered_universe)

    normalized_test_cases = normalize_test_cases(test_cases)

    for test_case in normalized_test_cases:
        if not test_case.issubset(universe_set):
            outside_elements = sorted(test_case - universe_set)
            raise ValueError(
                "A test case contains elements outside the universe: "
                f"{outside_elements}"
            )

    return ordered_universe, normalized_test_cases


def trivial_lower_bound(
    universe: Iterable[int],
    test_cases: Iterable[Iterable[int]],
) -> int:
    """
    Compute the trivial lower bound.

    If there is at least one test case, any hitting set must contain
    at least one element. If there are no test cases, the empty set is already
    a valid hitting set, so the lower bound is 0.

    Parameters
    ----------
    universe:
        Iterable containing all universe elements.
        This parameter is included for interface consistency.

    test_cases:
        Iterable containing the test cases / conflicts.

    Returns
    -------
    int
        0 if there are no test cases, otherwise 1.
    """
    _ = list(universe)
    test_cases = list(test_cases)

    return 1 if len(test_cases) > 0 else 0


def element_degrees(
    universe: Iterable[int],
    test_cases: Iterable[Iterable[int]],
) -> Dict[int, int]:
    """
    Count how many test cases are hit by each universe element.

    The degree of an element i is:

        deg(i) = number of test cases C_j such that i in C_j

    Parameters
    ----------
    universe:
        Iterable containing all universe elements.

    test_cases:
        Iterable containing the test cases / conflicts.

    Returns
    -------
    Dict[int, int]
        Dictionary mapping each universe element to its degree.
    """
    ordered_universe, normalized_test_cases = validate_instance(
        universe,
        test_cases,
    )

    degrees = {element: 0 for element in ordered_universe}

    for test_case in normalized_test_cases:
        for element in test_case:
            degrees[element] += 1

    return degrees


def max_degree_lower_bound(
    universe: Iterable[int],
    test_cases: Iterable[Iterable[int]],
) -> int:
    """
    Compute the maximum-degree lower bound.

    Let m be the number of test cases and let d_max be the maximum degree
    of any universe element.

    One selected element can hit at most d_max test cases. Therefore, to hit
    all m test cases we need at least:

        ceil(m / d_max)

    elements.

    Parameters
    ----------
    universe:
        Iterable containing all universe elements.

    test_cases:
        Iterable containing the test cases / conflicts.

    Returns
    -------
    int
        The maximum-degree lower bound.

    Raises
    ------
    ValueError
        If there are test cases but no element can hit them.
    """
    ordered_universe, normalized_test_cases = validate_instance(
        universe,
        test_cases,
    )

    if len(normalized_test_cases) == 0:
        return 0

    degrees = element_degrees(ordered_universe, normalized_test_cases)
    max_degree = max(degrees.values(), default=0)

    if max_degree == 0:
        raise ValueError(
            "The instance is infeasible: test cases exist, "
            "but no universe element appears in them."
        )

    return ceil(len(normalized_test_cases) / max_degree)


def sum_degree_lower_bound(
    universe: Iterable[int],
    test_cases: Iterable[Iterable[int]],
) -> int:
    """
    Compute the sum-degree lower bound.

    Sort all element degrees in decreasing order:

        d_1 >= d_2 >= ... >= d_n

    Even in the most optimistic case, r selected elements can hit at most:

        d_1 + d_2 + ... + d_r

    test cases.

    Therefore, the smallest r such that this cumulative sum reaches the
    number of test cases is a valid lower bound.

    Parameters
    ----------
    universe:
        Iterable containing all universe elements.

    test_cases:
        Iterable containing the test cases / conflicts.

    Returns
    -------
    int
        The sum-degree lower bound.

    Raises
    ------
    ValueError
        If the test cases cannot be covered by the universe.
    """
    ordered_universe, normalized_test_cases = validate_instance(
        universe,
        test_cases,
    )

    if len(normalized_test_cases) == 0:
        return 0

    degrees = sorted(
        element_degrees(ordered_universe, normalized_test_cases).values(),
        reverse=True,
    )

    cumulative = 0

    for r, degree in enumerate(degrees, start=1):
        cumulative += degree

        if cumulative >= len(normalized_test_cases):
            return r

    raise ValueError(
        "The instance is infeasible: the universe cannot cover all test cases."
    )


def greedy_packing_lower_bound(
    universe: Iterable[int],
    test_cases: Iterable[Iterable[int]],
) -> int:
    """
    Compute a greedy disjoint-packing lower bound.

    A collection of pairwise disjoint test cases gives a lower bound because
    a single selected element can hit at most one test case from that collection.

    This function greedily builds such a packing by scanning smaller test cases
    first.

    Parameters
    ----------
    universe:
        Iterable containing all universe elements.
        Used for validation.

    test_cases:
        Iterable containing the test cases / conflicts.

    Returns
    -------
    int
        Number of pairwise disjoint test cases found greedily.
    """
    _, normalized_test_cases = validate_instance(universe, test_cases)

    used_elements: Set[int] = set()
    packing_size = 0

    for test_case in sorted(
        normalized_test_cases,
        key=lambda test_case: (len(test_case), sorted(test_case)),
    ):
        if test_case.isdisjoint(used_elements):
            used_elements.update(test_case)
            packing_size += 1

    return packing_size


def lp_relaxation_lower_bound(
    universe: Iterable[int],
    test_cases: Iterable[Iterable[int]],
) -> int:
    """
    Compute the LP-relaxation lower bound.

    The integer minimum hitting set problem is:

        minimize    sum_i x_i
        subject to  sum_{i in C_j} x_i >= 1    for every test case C_j
                    x_i in {0, 1}

    The LP relaxation replaces:

        x_i in {0, 1}

    with:

        0 <= x_i <= 1

    Since the LP allows fractional solutions, its optimum is less than or
    equal to the integer optimum. Therefore:

        ceil(LP optimum)

    is a valid lower bound for the minimum hitting set size.

    Parameters
    ----------
    universe:
        Iterable containing all universe elements.

    test_cases:
        Iterable containing the test cases / conflicts.

    Returns
    -------
    int
        Ceil of the LP relaxation optimum.

    Raises
    ------
    ImportError
        If scipy is not installed.

    RuntimeError
        If the LP solver fails.
    """
    try:
        from scipy.optimize import linprog
    except ImportError as exc:
        raise ImportError(
            "scipy is required for lp_relaxation_lower_bound. "
            "Install it with: pip install scipy"
        ) from exc

    ordered_universe, normalized_test_cases = validate_instance(
        universe,
        test_cases,
    )

    if len(normalized_test_cases) == 0:
        return 0

    element_to_index = {
        element: idx
        for idx, element in enumerate(ordered_universe)
    }

    num_variables = len(ordered_universe)

    # Objective: minimize sum_i x_i.
    c = np.ones(num_variables)

    # scipy.linprog uses constraints of the form:
    #
    #     A_ub x <= b_ub
    #
    # The hitting constraints are:
    #
    #     sum_{i in C_j} x_i >= 1
    #
    # so we multiply by -1:
    #
    #     -sum_{i in C_j} x_i <= -1
    A_ub = []
    b_ub = []

    for test_case in normalized_test_cases:
        row = np.zeros(num_variables)

        for element in test_case:
            row[element_to_index[element]] = -1.0

        A_ub.append(row)
        b_ub.append(-1.0)

    bounds = [(0.0, 1.0) for _ in range(num_variables)]

    result = linprog(
        c=c,
        A_ub=np.array(A_ub),
        b_ub=np.array(b_ub),
        bounds=bounds,
        method="highs",
    )

    if not result.success:
        raise RuntimeError(f"LP relaxation failed: {result.message}")

    return ceil(float(result.fun) - 1.0e-9)


def compute_lower_bounds(
    universe: Iterable[int],
    test_cases: Iterable[Iterable[int]],
    use_lp: bool = False,
    prune_dominated: bool = False,
) -> LowerBoundResult:
    """
    Compute all selected lower bounds for a hitting set instance.

    Parameters
    ----------
    universe:
        Iterable containing all universe elements.

    test_cases:
        Iterable containing the test cases / conflicts.

    use_lp:
        If True, also compute the LP-relaxation lower bound.
        This is usually stronger, but requires scipy and is more expensive.

    prune_dominated:
        If True, remove duplicate and dominated test cases before computing
        the lower bounds.

        This is safe because if C_small is contained in C_big, then C_big is
        redundant for the hitting set constraints.

    Returns
    -------
    LowerBoundResult
        Dataclass containing all computed lower bounds.
    """
    ordered_universe, normalized_test_cases = validate_instance(
        universe,
        test_cases,
    )

    if prune_dominated:
        normalized_test_cases = remove_dominated_test_cases(normalized_test_cases)

    lp_bound = None

    if use_lp:
        lp_bound = lp_relaxation_lower_bound(
            ordered_universe,
            normalized_test_cases,
        )

    return LowerBoundResult(
        trivial=trivial_lower_bound(
            ordered_universe,
            normalized_test_cases,
        ),
        max_degree=max_degree_lower_bound(
            ordered_universe,
            normalized_test_cases,
        ),
        sum_degree=sum_degree_lower_bound(
            ordered_universe,
            normalized_test_cases,
        ),
        greedy_packing=greedy_packing_lower_bound(
            ordered_universe,
            normalized_test_cases,
        ),
        lp_relaxation=lp_bound,
    )


def greedy_upper_bound(
    universe: Iterable[int],
    test_cases: Iterable[Iterable[int]],
    prune_dominated: bool = False,
) -> Set[int]:
    """
    Compute a greedy hitting set.

    This gives an upper bound on the minimum hitting set size.

    The algorithm repeatedly selects the universe element that hits the largest
    number of currently unhit test cases.

    Parameters
    ----------
    universe:
        Iterable containing all universe elements.

    test_cases:
        Iterable containing the test cases / conflicts.

    prune_dominated:
        If True, remove duplicate and dominated test cases before running
        the greedy heuristic.

    Returns
    -------
    Set[int]
        A valid hitting set found greedily.

    Raises
    ------
    ValueError
        If the instance is infeasible.
    """
    ordered_universe, normalized_test_cases = validate_instance(
        universe,
        test_cases,
    )

    if prune_dominated:
        normalized_test_cases = remove_dominated_test_cases(normalized_test_cases)

    remaining = [set(test_case) for test_case in normalized_test_cases]
    solution: Set[int] = set()

    while remaining:
        best_element = None
        best_hits = -1

        for element in ordered_universe:
            hits = sum(
                1
                for test_case in remaining
                if element in test_case
            )

            if hits > best_hits:
                best_hits = hits
                best_element = element

        if best_element is None or best_hits <= 0:
            raise ValueError(
                "The instance is infeasible: remaining test cases cannot be hit."
            )

        solution.add(best_element)

        remaining = [
            test_case
            for test_case in remaining
            if best_element not in test_case
        ]

    return solution


@dataclass(frozen=True)
class PreprocessingResult:
    """
    Container for preprocessing information.

    Parameters
    ----------
    lower_bounds:
        Result containing all lower bounds.

    greedy_solution:
        Greedy hitting set used as an upper bound.

    num_test_cases_original:
        Number of test cases before optional pruning.

    num_test_cases_after_pruning:
        Number of test cases after optional dominated-test-case pruning.

    start_cardinality:
        First cardinality that should be tried in a linear sweep.

    stop_cardinality:
        Last cardinality that needs to be tried if you use the greedy upper
        bound as a stopping point.
    """

    lower_bounds: LowerBoundResult
    greedy_solution: Set[int]
    num_test_cases_original: int
    num_test_cases_after_pruning: int
    start_cardinality: int
    stop_cardinality: int

    def as_dict(self) -> Dict[str, object]:
        """
        Convert preprocessing result to a dictionary.

        Useful for storing this object inside algorithm metadata.

        Returns
        -------
        Dict[str, object]
            Dictionary representation of the preprocessing result.
        """
        return {
            "lower_bounds": self.lower_bounds.as_dict(),
            "greedy_solution": sorted(self.greedy_solution),
            "greedy_upper_bound": len(self.greedy_solution),
            "num_test_cases_original": self.num_test_cases_original,
            "num_test_cases_after_pruning": self.num_test_cases_after_pruning,
            "start_cardinality": self.start_cardinality,
            "stop_cardinality": self.stop_cardinality,
        }


def preprocess_hitting_set_instance(
    universe: Iterable[int],
    test_cases: Iterable[Iterable[int]],
    use_lp_lower_bound: bool = False,
    prune_dominated: bool = False,
) -> PreprocessingResult:
    """
    Run the full preprocessing pipeline for a hitting set instance.

    This computes:

    1. optional dominated-test-case pruning;
    2. several lower bounds;
    3. a greedy upper bound;
    4. recommended start and stop cardinalities for a cardinality sweep.

    Parameters
    ----------
    universe:
        Iterable containing all universe elements.

    test_cases:
        Iterable containing the test cases / conflicts.

    use_lp_lower_bound:
        If True, compute the LP-relaxation lower bound.
        This can be stronger but requires scipy.

    prune_dominated:
        If True, remove duplicate and dominated test cases before computing
        bounds.

    Returns
    -------
    PreprocessingResult
        Full preprocessing result.
    """
    ordered_universe, normalized_test_cases = validate_instance(
        universe,
        test_cases,
    )

    num_test_cases_original = len(normalized_test_cases)

    if prune_dominated:
        processed_test_cases = remove_dominated_test_cases(normalized_test_cases)
    else:
        processed_test_cases = normalized_test_cases

    lower_bounds = compute_lower_bounds(
        universe=ordered_universe,
        test_cases=processed_test_cases,
        use_lp=use_lp_lower_bound,
        prune_dominated=False,
    )

    greedy_solution = greedy_upper_bound(
        universe=ordered_universe,
        test_cases=processed_test_cases,
        prune_dominated=False,
    )

    start_cardinality = lower_bounds.value
    stop_cardinality = len(greedy_solution)

    return PreprocessingResult(
        lower_bounds=lower_bounds,
        greedy_solution=greedy_solution,
        num_test_cases_original=num_test_cases_original,
        num_test_cases_after_pruning=len(processed_test_cases),
        start_cardinality=start_cardinality,
        stop_cardinality=stop_cardinality,
    )

@dataclass(frozen=True)
class LengthOnePruningResult:
    """
    Container for length-one solution pruning.

    Parameters
    ----------
    pruned_universe:
        Ordered universe after removing the elements that appear in
        singleton hitting-set solutions.

    pruned_test_cases:
        Test cases after removing those singleton-solution elements.

    removed_elements:
        Elements removed from the universe and from the test cases.

    feasible_without_singletons:
        False if at least one test case becomes empty after pruning.
        In that case, the reduced instance has no hitting set that avoids
        the removed singleton-solution elements.
    """

    pruned_universe: List[int]
    pruned_test_cases: List[Set[int]]
    removed_elements: Set[int]
    feasible_without_singletons: bool

    def as_dict(self) -> Dict[str, object]:
        """
        Convert the pruning result to a dictionary.

        Returns
        -------
        Dict[str, object]
            Dictionary representation of the pruning result.
        """
        return {
            "pruned_universe": self.pruned_universe,
            "pruned_test_cases": [
                sorted(test_case)
                for test_case in self.pruned_test_cases
            ],
            "removed_elements": sorted(self.removed_elements),
            "num_removed_elements": len(self.removed_elements),
            "feasible_without_singletons": self.feasible_without_singletons,
        }
    
def prune_length_one_solutions(
    universe: Iterable[int],
    test_cases: Iterable[Iterable[int]],
    singleton_solutions: Iterable[Iterable[int]],
) -> LengthOnePruningResult:
    """
    Remove all elements that appear in known length-one hitting-set solutions.

    This function is meant to be used after a classical preprocessing step
    has already found all singleton hitting sets.

    A singleton solution should be represented as an iterable containing
    exactly one element, for example:

        {3}

    or:

        frozenset({3})

    If singleton_solutions contains {3}, then element 3 is removed from the
    universe and from every test case.

    Parameters
    ----------
    universe:
        Iterable containing all universe elements.

    test_cases:
        Iterable containing the test cases / conflicts.

    singleton_solutions:
        Iterable containing the known length-one hitting-set solutions.
        Each solution must contain exactly one element.

    Returns
    -------
    LengthOnePruningResult
        Full pruning result.

    Raises
    ------
    ValueError
        If one of the provided singleton solutions does not contain exactly
        one element.

    ValueError
        If one of the singleton-solution elements does not belong to the
        universe.
    """
    ordered_universe, normalized_test_cases = validate_instance(
        universe,
        test_cases,
    )

    universe_set = set(ordered_universe)

    removed_elements: Set[int] = set()

    for singleton_solution in singleton_solutions:
        singleton_set = set(singleton_solution)

        if len(singleton_set) != 1:
            raise ValueError(
                "Every length-one solution must contain exactly one element. "
                f"Received: {sorted(singleton_set)}"
            )

        element = next(iter(singleton_set))

        if element not in universe_set:
            raise ValueError(
                "A singleton-solution element does not belong to the universe: "
                f"{element}"
            )

        removed_elements.add(element)

    if len(removed_elements) == 0:
        return LengthOnePruningResult(
            pruned_universe=ordered_universe,
            pruned_test_cases=normalized_test_cases,
            removed_elements=set(),
            feasible_without_singletons=True,
        )

    pruned_universe = [
        element
        for element in ordered_universe
        if element not in removed_elements
    ]

    pruned_test_cases = [
        set(test_case) - removed_elements
        for test_case in normalized_test_cases
    ]

    feasible_without_singletons = all(
        len(test_case) > 0
        for test_case in pruned_test_cases
    )

    return LengthOnePruningResult(
        pruned_universe=pruned_universe,
        pruned_test_cases=pruned_test_cases,
        removed_elements=removed_elements,
        feasible_without_singletons=feasible_without_singletons,
    )