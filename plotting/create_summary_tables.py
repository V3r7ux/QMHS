from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Algorithms and labels used in the paper.
# -----------------------------------------------------------------------------

RQ1_ALGORITHMS = [
    "CardinalitySweep",
    "RecursivePUBO",
    "RecursivePUBO With Preprocessing and Adaptive Qubits",
]

RQ1_LABELS = {
    "CardinalitySweep": "Cardinality Sweep",
    "RecursivePUBO": "Recursive PUBO",
    "RecursivePUBO With Preprocessing and Adaptive Qubits": "Recursive PUBO + all opt.",
}

RQ2_RQ3_ALGORITHMS = [
    "LinearSearch",
    "LinearSearch With Preprocessing and Reduced Qubits",
    "BinSearch",
]

RQ2_RQ3_LABELS = {
    "LinearSearch": "LinearSearch",
    "LinearSearch With Preprocessing and Reduced Qubits": "Linear + all opt.",
    "BinSearch": "BinSearch + all opt.",
}

EPS = 1e-12


# -----------------------------------------------------------------------------
# Generic utilities.
# -----------------------------------------------------------------------------

def require_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{name} is missing required columns: {sorted(missing)}. "
            f"Available columns are: {sorted(df.columns)}"
        )


def parse_activation_probability(problem_id: object) -> float | None:
    if pd.isna(problem_id):
        return None

    match = re.search(r"(?:^|-)pa-([0-9]+(?:\.[0-9]+)?)(?:-|$)", str(problem_id))
    if match is None:
        return None
    return float(match.group(1))


def check_algorithms(df: pd.DataFrame, algorithms: list[str], name: str) -> None:
    available = set(df["algorithm"].dropna().unique().tolist())
    missing = [algorithm for algorithm in algorithms if algorithm not in available]
    if missing:
        raise ValueError(
            f"{name}: missing required algorithms {missing}. "
            f"Available algorithms: {sorted(available)}"
        )


def finite_numeric(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    return values


def format_number(value: float, decimals: int = 3) -> str:
    if pd.isna(value):
        return "--"
    if decimals == 0:
        return str(int(round(float(value))))
    return f"{float(value):.{decimals}f}"


def format_median_range(
    series: pd.Series,
    decimals: int = 3,
) -> str:
    values = finite_numeric(series)
    if values.empty:
        return "--"

    return (
        f"{format_number(values.median(), decimals)} "
        f"[{format_number(values.min(), decimals)}, "
        f"{format_number(values.max(), decimals)}]"
    )


def format_mean(
    series: pd.Series,
    decimals: int = 3,
) -> str:
    values = finite_numeric(series)
    if values.empty:
        return "--"

    return format_number(values.mean(), decimals)


def format_median(
    series: pd.Series,
    decimals: int = 3,
) -> str:
    values = finite_numeric(series)
    if values.empty:
        return "--"

    return format_number(values.median(), decimals)


def format_count_percentage(mask: pd.Series) -> str:
    mask = mask.fillna(False).astype(bool)
    n = int(mask.sum())
    total = int(len(mask))
    pct = 100.0 * n / total if total else np.nan
    return f"{n}/{total} ({pct:.1f}%)" if total else "--"


def escape_latex(text: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def dataframe_to_latex_table(
    table: pd.DataFrame,
    caption: str,
    label: str,
) -> str:
    """Create compact booktabs LaTeX for an already-formatted wide table."""
    columns = list(table.columns)
    alignment = "l" + "c" * (len(columns) - 1)

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\small",
        f"\\begin{{tabular}}{{{alignment}}}",
        r"\toprule",
        " & ".join(escape_latex(str(c)) for c in columns) + r" \\",
        r"\midrule",
    ]

    for _, row in table.iterrows():
        lines.append(
            " & ".join(escape_latex(str(row[col])) for col in columns) + r" \\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ])

    return "\n".join(lines)


def save_table(
    table: pd.DataFrame,
    output_dir: Path,
    stem: str,
    caption: str,
    label: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{stem}.csv"
    tex_path = output_dir / f"{stem}.tex"

    table.to_csv(csv_path, index=False)
    tex_path.write_text(
        dataframe_to_latex_table(table, caption=caption, label=label),
        encoding="utf-8",
    )

    print(f"Saved: {csv_path}")
    print(f"Saved: {tex_path}")


# -----------------------------------------------------------------------------
# RQ1 / RQ2 loading and subject-level normalization.
# -----------------------------------------------------------------------------

def load_global_stats(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    require_columns(
        df,
        {
            "problem_id",
            "algorithm",
            "universe_size",
            "num_test_cases",
            "num_qubits",
            "num_couplers",
            "global_found_ratio",
            "runtime_seconds",
        },
        name="global_stats.csv",
    )

    df = df.copy()

    if "activation_probability" not in df.columns:
        df["activation_probability"] = df["problem_id"].map(parse_activation_probability)

    return df


def collapse_global_repetitions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure that each benchmark subject contributes exactly once per algorithm.

    If global_stats.csv already contains one row per problem_id/algorithm pair,
    this function leaves the values unchanged. If repeated runs are present,
    continuous quantities are averaged first so repetitions do not overweight a
    subject in the final summary table.
    """
    keys = [
        "problem_id",
        "algorithm",
        "universe_size",
        "num_test_cases",
        "activation_probability",
    ]

    duplicated = df.duplicated(subset=["problem_id", "algorithm"], keep=False)
    if duplicated.any():
        print(
            "Warning: repeated problem_id/algorithm rows were found in global_stats.csv. "
            "They will be averaged before computing the subject-level summaries."
        )

    agg_spec: dict[str, tuple[str, str]] = {
        "global_found_ratio": ("global_found_ratio", "mean"),
        "runtime_seconds": ("runtime_seconds", "mean"),
        "num_qubits": ("num_qubits", "mean"),
        "num_couplers": ("num_couplers", "mean"),
    }

    if "total_true_mhs" in df.columns:
        agg_spec["total_true_mhs"] = ("total_true_mhs", "max")
    if "total_found_mhs" in df.columns:
        agg_spec["total_found_mhs"] = ("total_found_mhs", "mean")

    return (
        df.groupby(keys, as_index=False, dropna=False)
        .agg(**agg_spec)
        .sort_values(["algorithm", "problem_id"])
        .reset_index(drop=True)
    )


# -----------------------------------------------------------------------------
# RQ1 table.
# -----------------------------------------------------------------------------

def build_rq1_table(path: str | Path) -> pd.DataFrame:
    df = collapse_global_repetitions(load_global_stats(path))
    df = df.loc[df["algorithm"].isin(RQ1_ALGORITHMS)].copy()
    check_algorithms(df, RQ1_ALGORITHMS, "RQ1")

    rows: list[dict[str, str]] = []

    metric_builders = [
        (
            "Full recovery, $n/N$ (%)",
            lambda g: format_count_percentage(g["global_found_ratio"] >= 1.0 - EPS),
        ),
        (
            "Partial recovery, $n/N$ (%)",
            lambda g: format_count_percentage(
                (g["global_found_ratio"] > EPS)
                & (g["global_found_ratio"] < 1.0 - EPS)
            ),
        ),
        (
            "Zero recovery, $n/N$ (%)",
            lambda g: format_count_percentage(g["global_found_ratio"] <= EPS),
        ),
        (
            "Mean found ratio",
            lambda g: format_mean(g["global_found_ratio"], decimals=3),
        ),
        (
            "Median found ratio",
            lambda g: format_median(g["global_found_ratio"], decimals=3),
        ),
        (
            "Runtime [s]",
            lambda g: format_median_range(g["runtime_seconds"], decimals=3),
        ),
        (
            "Logical variables",
            lambda g: format_median_range(g["num_qubits"], decimals=0),
        ),
        (
            "Logical couplers",
            lambda g: format_median_range(g["num_couplers"], decimals=0),
        ),
    ]

    groups = {algorithm: df.loc[df["algorithm"] == algorithm] for algorithm in RQ1_ALGORITHMS}

    for metric_name, builder in metric_builders:
        row: dict[str, str] = {"Metric": metric_name}
        for algorithm in RQ1_ALGORITHMS:
            row[RQ1_LABELS[algorithm]] = builder(groups[algorithm])
        rows.append(row)

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# RQ2 table.
# -----------------------------------------------------------------------------

def build_rq2_table(path: str | Path) -> pd.DataFrame:
    df = collapse_global_repetitions(load_global_stats(path))
    df = df.loc[df["algorithm"].isin(RQ2_RQ3_ALGORITHMS)].copy()
    check_algorithms(df, RQ2_RQ3_ALGORITHMS, "RQ2")

    # In RQ2, global_found_ratio is defined over the true minimum-cardinality
    # hitting sets. Therefore >0 means that at least one true minimum solution
    # was found, while ==1 means complete minimum-set enumeration.
    rows: list[dict[str, str]] = []

    metric_builders = [
        (
            "Minimum-cardinality success, $n/N$ (%)",
            lambda g: format_count_percentage(g["global_found_ratio"] > EPS),
        ),
        (
            "Full minimum-set recovery, $n/N$ (%)",
            lambda g: format_count_percentage(g["global_found_ratio"] >= 1.0 - EPS),
        ),
        (
            "Optimization success but incomplete enumeration, $n/N$ (%)",
            lambda g: format_count_percentage(
                (g["global_found_ratio"] > EPS)
                & (g["global_found_ratio"] < 1.0 - EPS)
            ),
        ),
        (
            "Zero minimum-set recovery, $n/N$ (%)",
            lambda g: format_count_percentage(g["global_found_ratio"] <= EPS),
        ),
        (
            "Mean minimum-set found ratio",
            lambda g: format_mean(g["global_found_ratio"], decimals=3),
        ),
        (
            "Median minimum-set found ratio",
            lambda g: format_median(g["global_found_ratio"], decimals=3),
        ),
        (
            "Runtime [s]",
            lambda g: format_median_range(g["runtime_seconds"], decimals=3),
        ),
        (
            "Logical variables",
            lambda g: format_median_range(g["num_qubits"], decimals=0),
        ),
        (
            "Logical couplers",
            lambda g: format_median_range(g["num_couplers"], decimals=0),
        ),
    ]

    groups = {
        algorithm: df.loc[df["algorithm"] == algorithm]
        for algorithm in RQ2_RQ3_ALGORITHMS
    }

    for metric_name, builder in metric_builders:
        row: dict[str, str] = {"Metric": metric_name}
        for algorithm in RQ2_RQ3_ALGORITHMS:
            row[RQ2_RQ3_LABELS[algorithm]] = builder(groups[algorithm])
        rows.append(row)

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# RQ3 loading / aggregation.
# Mirrors the aggregation hierarchy used by main_plots_rq3.py, but stops at
# one row per benchmark subject and algorithm instead of averaging over p_a.
# -----------------------------------------------------------------------------

def load_embedding_attempts(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    require_columns(
        df,
        {
            "problem_id",
            "algorithm",
            "universe_size",
            "num_test_cases",
            "attempt_id",
            "embedding_found",
            "runtime_seconds",
            "num_physical_qubits",
            "max_chain_length",
            "mean_chain_length",
            "topology",
            "topology_size",
            "target_cardinality",
        },
        name="embedding_attempts.csv",
    )

    df = df.copy()
    if "activation_probability" not in df.columns:
        df["activation_probability"] = df["problem_id"].map(parse_activation_probability)

    # Robust conversion for CSVs where booleans are stored as strings.
    if df["embedding_found"].dtype == object:
        normalized = df["embedding_found"].astype(str).str.strip().str.lower()
        mapping = {"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0}
        df["embedding_found"] = normalized.map(mapping)
    else:
        df["embedding_found"] = pd.to_numeric(df["embedding_found"], errors="coerce")

    if df["embedding_found"].isna().any():
        raise ValueError("Could not convert some embedding_found values to 0/1.")

    return df


def select_topology(df: pd.DataFrame, topology: str | None) -> pd.DataFrame:
    available = sorted(df["topology"].dropna().astype(str).unique().tolist())

    if topology is None:
        if len(available) > 1:
            raise ValueError(
                "RQ3 data contains multiple topologies. Pass --rq3-topology with one of: "
                f"{available}"
            )
        return df

    mask = df["topology"].astype(str).str.casefold() == topology.casefold()
    selected = df.loc[mask].copy()
    if selected.empty:
        raise ValueError(
            f"No RQ3 rows matched topology {topology!r}. Available topologies: {available}"
        )
    return selected


def aggregate_attempts_to_submissions(attempts: pd.DataFrame) -> pd.DataFrame:
    optional_submission_keys = [
        col for col in ["iteration", "stage"] if col in attempts.columns
    ]

    submission_keys = [
        "problem_id",
        "algorithm",
        "universe_size",
        "num_test_cases",
        "activation_probability",
        "topology",
        "topology_size",
        "target_cardinality",
        *optional_submission_keys,
    ]

    submission = (
        attempts.groupby(submission_keys, as_index=False, dropna=False)
        .agg(
            embedding_success=("embedding_found", "mean"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
            mean_num_physical_qubits=("num_physical_qubits", "mean"),
            mean_chain_length=("mean_chain_length", "mean"),
            max_chain_length=("max_chain_length", "mean"),
            attempts_per_submission=("attempt_id", "nunique"),
        )
    )

    return submission.sort_values(submission_keys).reset_index(drop=True)


def aggregate_submissions_to_instances(submissions: pd.DataFrame) -> pd.DataFrame:
    instance_keys = [
        "problem_id",
        "algorithm",
        "universe_size",
        "num_test_cases",
        "activation_probability",
        "topology",
        "topology_size",
    ]

    instance = (
        submissions.groupby(instance_keys, as_index=False, dropna=False)
        .agg(
            embedding_success=("embedding_success", "mean"),
            mean_chain_length=("mean_chain_length", "mean"),
            max_chain_length=("max_chain_length", "mean"),
            mean_num_physical_qubits=("mean_num_physical_qubits", "mean"),
            runtime_seconds=("mean_runtime_seconds", "sum"),
            num_embedding_calls=("target_cardinality", "count"),
            mean_attempts_per_submission=("attempts_per_submission", "mean"),
        )
    )

    return instance.sort_values(instance_keys).reset_index(drop=True)


def load_embedding_stats_for_ratio(path: str | Path) -> pd.DataFrame:
    stats = pd.read_csv(path)

    if "activation_probability" not in stats.columns:
        stats["activation_probability"] = stats["problem_id"].map(parse_activation_probability)

    require_columns(
        stats,
        {
            "problem_id",
            "algorithm",
            "universe_size",
            "num_test_cases",
            "activation_probability",
            "topology",
            "topology_size",
            "target_cardinality",
            "physical_logical_qubit_ratio",
        },
        name="embedding_stats.csv",
    )

    return stats


def add_physical_logical_ratio_to_instances(
    instances: pd.DataFrame,
    stats_path: str | Path,
    topology: str | None,
) -> pd.DataFrame:
    stats = select_topology(load_embedding_stats_for_ratio(stats_path), topology)

    optional_keys = [col for col in ["iteration", "stage"] if col in stats.columns]
    submission_keys = [
        "problem_id",
        "algorithm",
        "universe_size",
        "num_test_cases",
        "activation_probability",
        "topology",
        "topology_size",
        "target_cardinality",
        *optional_keys,
    ]

    ratio_submissions = (
        stats.groupby(submission_keys, as_index=False, dropna=False)
        .agg(physical_logical_qubit_ratio=("physical_logical_qubit_ratio", "mean"))
    )

    instance_keys = [
        "problem_id",
        "algorithm",
        "universe_size",
        "num_test_cases",
        "activation_probability",
        "topology",
        "topology_size",
    ]

    ratio_instances = (
        ratio_submissions.groupby(instance_keys, as_index=False, dropna=False)
        .agg(physical_logical_qubit_ratio=("physical_logical_qubit_ratio", "mean"))
    )

    return instances.merge(ratio_instances, on=instance_keys, how="left")


def collapse_hitman_runtime_per_problem(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path).copy()

    if "problem_name" in df.columns and "problem_id" not in df.columns:
        df = df.rename(columns={"problem_name": "problem_id"})

    require_columns(df, {"problem_id", "runtime_seconds"}, "Hitman CSV")

    return (
        df.groupby("problem_id", as_index=False, dropna=False)
        .agg(hitman_runtime_seconds=("runtime_seconds", "mean"))
    )


def add_hitman_ratio_to_instances(
    instances: pd.DataFrame,
    hitman_path: str | Path,
) -> pd.DataFrame:
    hitman = collapse_hitman_runtime_per_problem(hitman_path)
    merged = instances.merge(hitman, on="problem_id", how="left")

    merged["embedding_over_hitman"] = (
        pd.to_numeric(merged["runtime_seconds"], errors="coerce")
        / (pd.to_numeric(merged["hitman_runtime_seconds"], errors="coerce") + EPS)
    )

    return merged


def build_rq3_table(
    attempts_path: str | Path,
    stats_path: str | Path | None,
    hitman_path: str | Path | None,
    topology: str | None,
) -> pd.DataFrame:
    attempts = select_topology(load_embedding_attempts(attempts_path), topology)
    attempts = attempts.loc[attempts["algorithm"].isin(RQ2_RQ3_ALGORITHMS)].copy()
    check_algorithms(attempts, RQ2_RQ3_ALGORITHMS, "RQ3")

    submissions = aggregate_attempts_to_submissions(attempts)
    instances = aggregate_submissions_to_instances(submissions)

    if stats_path is not None:
        instances = add_physical_logical_ratio_to_instances(
            instances=instances,
            stats_path=stats_path,
            topology=topology,
        )

    if hitman_path is not None:
        instances = add_hitman_ratio_to_instances(instances, hitman_path)

    groups = {
        algorithm: instances.loc[instances["algorithm"] == algorithm]
        for algorithm in RQ2_RQ3_ALGORITHMS
    }

    metrics: list[tuple[str, str, int]] = [
        ("Embedding success rate", "embedding_success", 3),
        ("Embedding calls", "num_embedding_calls", 0),
        ("Embedding runtime [s]", "runtime_seconds", 3),
        ("Physical qubits", "mean_num_physical_qubits", 1),
        ("Mean chain length", "mean_chain_length", 3),
        ("Maximum chain length", "max_chain_length", 3),
    ]

    if "physical_logical_qubit_ratio" in instances.columns:
        metrics.insert(
            4,
            ("Physical/logical qubit ratio", "physical_logical_qubit_ratio", 3),
        )

    if "embedding_over_hitman" in instances.columns:
        metrics.append(
            ("Embedding/Hitman runtime ratio", "embedding_over_hitman", 2)
        )

    rows: list[dict[str, str]] = []

    for metric_name, column, decimals in metrics:
        row: dict[str, str] = {"Metric": metric_name}
        for algorithm in RQ2_RQ3_ALGORITHMS:
            row[RQ2_RQ3_LABELS[algorithm]] = format_median_range(
                groups[algorithm][column],
                decimals=decimals,
            )
        rows.append(row)

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Main CLI.
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the three paper summary tables for RQ1, RQ2, and RQ3. "
            "Continuous metrics are reported as median [min, max]. Recovery "
            "metrics are reported as n/N (percentage)."
        )
    )

    parser.add_argument("--rq1-global", required=True, type=Path,
                        help="RQ1 global_stats.csv")
    parser.add_argument("--rq2-global", required=True, type=Path,
                        help="RQ2 global_stats.csv")
    parser.add_argument("--rq3-attempts", required=True, type=Path,
                        help="RQ3 embedding_attempts.csv")
    parser.add_argument("--rq3-stats", type=Path, default=None,
                        help="RQ3 embedding_stats.csv; required for physical/logical ratio")
    parser.add_argument("--hitman", type=Path, default=None,
                        help="Hitman/classical runtime CSV; required for embedding/Hitman ratio")
    parser.add_argument("--rq3-topology", type=str, default=None,
                        help="Topology to summarize, e.g. zephyr. Required if the RQ3 CSV contains multiple topologies.")
    parser.add_argument("--output-dir", required=True, type=Path)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rq1 = build_rq1_table(args.rq1_global)
    rq2 = build_rq2_table(args.rq2_global)
    rq3 = build_rq3_table(
        attempts_path=args.rq3_attempts,
        stats_path=args.rq3_stats,
        hitman_path=args.hitman,
        topology=args.rq3_topology,
    )

    save_table(
        rq1,
        output_dir,
        stem="table_rq1_summary",
        caption=(
            "RQ1 summary across individual benchmark subjects. Continuous metrics "
            "are reported as median [minimum, maximum]."
        ),
        label="tab:rq1_summary",
    )

    save_table(
        rq2,
        output_dir,
        stem="table_rq2_summary",
        caption=(
            "RQ2 summary across individual benchmark subjects. Continuous metrics "
            "are reported as median [minimum, maximum]."
        ),
        label="tab:rq2_summary",
    )

    save_table(
        rq3,
        output_dir,
        stem="table_rq3_summary",
        caption=(
            "RQ3 summary across individual benchmark subjects. Metrics are reported "
            "as median [minimum, maximum]."
        ),
        label="tab:rq3_summary",
    )

    print("\nRQ1 table")
    print("=========")
    print(rq1.to_string(index=False))

    print("\nRQ2 table")
    print("=========")
    print(rq2.to_string(index=False))

    print("\nRQ3 table")
    print("=========")
    print(rq3.to_string(index=False))


if __name__ == "__main__":
    main()
