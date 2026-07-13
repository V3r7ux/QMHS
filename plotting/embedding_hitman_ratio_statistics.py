from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------
# Configuration.
# ---------------------------------------------------------------------

RATIO_COLUMN = "mean_embedding_over_hitman"

ALGORITHMS = [
    "LinearSearch",
    "LinearSearch With Preprocessing and Reduced Qubits",
    "BinSearch",
]

ALGORITHM_LABELS = {
    "LinearSearch": "LinearSearch",
    "LinearSearch With Preprocessing and Reduced Qubits": "Linear + all opt.",
    "BinSearch": "BinSearch + all opt.",
}


# ---------------------------------------------------------------------
# Loading and validation.
# ---------------------------------------------------------------------

def require_columns(
    df: pd.DataFrame,
    required: set[str],
    name: str = "df",
) -> None:
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{name} is missing required columns: {sorted(missing)}. "
            f"Available columns are: {sorted(df.columns)}"
        )


def load_hitman_ratio_heatmap_values(
    path: str | Path,
) -> pd.DataFrame:
    """
    Load the aggregated embedding/Hitman runtime ratio table and keep
    only the three algorithms used in the main RQ3 comparison.
    """
    df = pd.read_csv(path)

    require_columns(
        df,
        {
            "algorithm",
            "universe_size",
            "num_test_cases",
            RATIO_COLUMN,
        },
        name="embedding_hitman_ratio_heatmap_values_all_pa.csv",
    )

    df = df.copy()

    df[RATIO_COLUMN] = pd.to_numeric(
        df[RATIO_COLUMN],
        errors="coerce",
    )

    df = df.loc[
        df["algorithm"].isin(ALGORITHMS)
    ].copy()

    available = set(df["algorithm"].dropna().unique())
    missing_algorithms = [
        algorithm
        for algorithm in ALGORITHMS
        if algorithm not in available
    ]

    if missing_algorithms:
        raise ValueError(
            f"Missing required algorithms: {missing_algorithms}. "
            f"Available algorithms: {sorted(available)}"
        )

    missing_values = df[RATIO_COLUMN].isna().sum()

    if missing_values > 0:
        print(
            f"Warning: {missing_values} rows have no finite embedding/Hitman "
            "runtime ratio and will be ignored."
        )

    return (
        df
        .dropna(subset=[RATIO_COLUMN])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------
# Statistics.
# ---------------------------------------------------------------------

def calculate_ratio_statistics(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate minimum, median, and maximum embedding/Hitman runtime
    ratio for each selected algorithm.
    """
    stats = (
        df
        .groupby(
            "algorithm",
            as_index=False,
            dropna=False,
        )
        .agg(
            min_ratio=(RATIO_COLUMN, "min"),
            median_ratio=(RATIO_COLUMN, "median"),
            max_ratio=(RATIO_COLUMN, "max"),
            count=(RATIO_COLUMN, "count"),
        )
    )

    stats["algorithm"] = stats["algorithm"].map(ALGORITHM_LABELS)

    order = [
        ALGORITHM_LABELS[algorithm]
        for algorithm in ALGORITHMS
    ]

    stats["algorithm"] = pd.Categorical(
        stats["algorithm"],
        categories=order,
        ordered=True,
    )

    return (
        stats
        .sort_values("algorithm")
        .reset_index(drop=True)
    )


def calculate_global_ratio_statistics(
    df: pd.DataFrame,
) -> pd.Series:
    """
    Calculate global statistics using only the three selected algorithms.
    """
    ratio = df[RATIO_COLUMN]

    return pd.Series(
        {
            "min_ratio": ratio.min(),
            "median_ratio": ratio.median(),
            "max_ratio": ratio.max(),
            "count": ratio.count(),
        }
    )


# ---------------------------------------------------------------------
# Main run.
# ---------------------------------------------------------------------

def run(
    input_path: str | Path,
    output_path: str | Path | None = None,
) -> None:
    df = load_hitman_ratio_heatmap_values(input_path)

    algorithm_stats = calculate_ratio_statistics(df)
    global_stats = calculate_global_ratio_statistics(df)

    print("\nEmbedding / Hitman runtime ratio statistics")
    print("===========================================")

    print("\nGlobal statistics:")
    print(f"Minimum ratio : {global_stats['min_ratio']:.6f}")
    print(f"Median ratio  : {global_stats['median_ratio']:.6f}")
    print(f"Maximum ratio : {global_stats['max_ratio']:.6f}")
    print(f"Count         : {int(global_stats['count'])}")

    print("\nStatistics by algorithm:")
    print(
        algorithm_stats.to_string(
            index=False,
        )
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        algorithm_stats.to_csv(
            output_path,
            index=False,
        )

        print(f"\nSaved ratio statistics to: {output_path}")


# ---------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate minimum, median, and maximum embedding-over-Hitman "
            "runtime ratio for LinearSearch, LinearSearch with all "
            "optimizations, and BinSearch."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help=(
            "Path to "
            "embedding_hitman_ratio_heatmap_values_all_pa.csv."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output CSV path.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run(
        input_path=args.input,
        output_path=args.output,
    )