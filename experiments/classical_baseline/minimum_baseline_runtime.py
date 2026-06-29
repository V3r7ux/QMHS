from __future__ import annotations

import csv
from pathlib import Path

from tqdm import tqdm

from src.utils.parsing import parse_mhs2_problem
from src.algorithms.classical.hitman_minimum import HitmanMinimumOPT


def main() -> None:
    universe_sizes = [5, 10, 25, 50]
    num_test_cases = 50

    output_path = Path("hitman_minimum_runtime.csv")

    algorithm = HitmanMinimumOPT()

    rows = []

    for universe_size in universe_sizes:
        problems = parse_mhs2_problem(
            universe_size,
            num_test_cases,
            True,
        )

        for problem in tqdm(problems, desc=f"Universe size {universe_size}"):
            _, runtime = algorithm.run(problem)

            rows.append(
                {
                    "problem_name": problem.name,
                    "universe_size": problem.size,
                    "num_test_cases": len(problem.test_cases),
                    "runtime_seconds": runtime,
                }
            )

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "problem_name",
                "universe_size",
                "num_test_cases",
                "runtime_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()