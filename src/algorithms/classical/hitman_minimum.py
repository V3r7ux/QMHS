from __future__ import annotations

import time
from typing import List, Set, Tuple, FrozenSet, Any

from pysat.examples.hitman import Hitman

from src.algorithms.classical.base_algorithm import BaseAlgorithm
from src.core.problem import HittingSetProblem
from src.utils.validation import is_minimal_hitting_set


class HitmanMinimumOPT(BaseAlgorithm):
    """
    Enumerate all minimum-cardinality hitting sets using PySAT's Hitman.

    This algorithm enumerates hitting sets in non-decreasing cardinality order.
    It stops as soon as Hitman starts returning hitting sets larger than the
    first optimum size.

    Therefore, the returned list contains all minimum hitting sets, not all
    inclusion-minimal hitting sets.

    Returns
    -------
    tuple[list[set], float]
        A pair:

            (minimum_hitting_sets, runtime_seconds)

        where minimum_hitting_sets is a list of sets.
    """

    def run(self, problem: HittingSetProblem) -> Tuple[List[Set[Any]], float]:
        start_time = time.perf_counter()

        test_cases = [set(tc) for tc in problem.test_cases]

        # Empty family: the empty set is the unique minimum hitting set.
        if not test_cases:
            runtime = time.perf_counter() - start_time
            return [set()], runtime

        # If one test case is empty, no hitting set exists.
        # An empty conflict cannot be hit by any candidate.
        if any(len(tc) == 0 for tc in test_cases):
            runtime = time.perf_counter() - start_time
            return [], runtime

        minimum_hitting_sets: List[Set[Any]] = []
        optimum_size: int | None = None

        # htype="sorted" makes Hitman enumerate hitting sets in increasing size.
        # This is the key point for minimum-cardinality enumeration.
        with Hitman(bootstrap_with=test_cases, htype="sorted") as hitman:
            while True:
                hs = hitman.get()

                if hs is None:
                    break

                candidate = set(hs)
                candidate_size = len(candidate)

                if optimum_size is None:
                    optimum_size = candidate_size

                # Since htype="sorted" enumerates in non-decreasing cardinality,
                # once we see a larger solution, all remaining ones are larger too.
                if candidate_size > optimum_size:
                    break

                # Safety check: for a minimum hitting set, minimality follows
                # automatically, but this keeps the output consistent with your
                # existing validation logic.
                if is_minimal_hitting_set(candidate, test_cases):
                    minimum_hitting_sets.append(candidate)

                # Block exactly this hitting set, so Hitman can return the next one.
                hitman.block(hs)

        runtime = time.perf_counter() - start_time
        return minimum_hitting_sets, runtime