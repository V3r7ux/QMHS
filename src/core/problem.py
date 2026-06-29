from dataclasses import dataclass
from typing import List, Set

@dataclass
class HittingSetProblem:
    """
    HittingSetProblem represents a minimal hitting set problem instance.

    It stores the universe of available elements, the collection of test cases
    that must be hit, and the known minimal hitting set solutions for the problem.
    An optional name can be assigned to identify the instance.

    The class provides two properties:
    - size: returns the number of elements in the universe
    - num_tests: returns the number of test cases
    """
    universe: List[int]
    test_cases: List[Set[int]]
    minimal_hitting_sets: List[Set[int]]
    name: str = "unnamed"

    @property
    def size(self) -> int:
        return len(self.universe)
    
    @property
    def num_tests(self) -> int:
        return len(self.test_cases)