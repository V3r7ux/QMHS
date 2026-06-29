import numpy as np
from pathlib import Path
from src.core.problem import HittingSetProblem
import os

REPO_ROOT = Path(__file__).resolve().parents[2]

def parse_mhs2_problem(universe_size: int, max_test_cases: int, known_solutions_only: bool = False) -> list[HittingSetProblem]:
    """
    Parse a hitting set problem instance from the corresponding input files.

    The function reads the conflict sets and the known minimal hitting sets
    associated with the given problem identifier, reconstructs the universe
    of elements appearing in the conflicts, and returns a HittingSetProblem
    object containing the full problem description.

    The expected files are located in the data/spectras directory and follow
    the naming convention:
    - <problem_id>-conflicts.txt
    - <problem_id>-mhs.txt
    """

    # Parse conflicts
    conflicts_filepath = REPO_ROOT / 'data' / 'spectrasMHS2'
    problem_files = list(conflicts_filepath.glob(f"M-{universe_size}-*.txt"))
    problems = []
    for problem_file in problem_files:
        collection = []
        minimal_hitting_sets = []
        problem_name = problem_file.parts[-1].split('.spectra')[0]
        mhs_filepath = REPO_ROOT / 'data' / 'mhs2' / (str(problem_name)+'.mhs.txt')
        if (not os.path.exists(mhs_filepath)) and known_solutions_only:
            continue
        with problem_file.open('r') as f:
            header = f.readline().split(' ')
            universe_size = int(header[0])
            num_test_cases = int(header[1])
            if num_test_cases > max_test_cases:
                continue
            for line in f:
                subset = set(i+1 for i, b in enumerate(line.split(' ')[:universe_size]) if b == '1')
                if subset not in collection:
                    collection.append(subset)
        if os.path.exists(mhs_filepath):
            with mhs_filepath.open('r') as f:
                for line in f:
                    mhs = [int(x) for x in line.split(' ')[:-1]]
                    if len(mhs) > 0:
                        minimal_hitting_sets.append(set(mhs))
    
        #universe = list(range(1, max(list(set.union(*collection)))+1))
        universe = list(set.union(*collection))
        universe.sort()

        problems.append(
            HittingSetProblem(
                universe=universe,
                test_cases=collection,
                minimal_hitting_sets=minimal_hitting_sets,
                name=problem_name
            )
        )

    return problems