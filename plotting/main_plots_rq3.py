from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap

try:
    from scipy.interpolate import griddata
    from scipy.ndimage import gaussian_filter
except ImportError:
    griddata = None
    gaussian_filter = None


# ---------------------------------------------------------------------
# Algorithms for the main RQ3 comparison.
# ---------------------------------------------------------------------

BASELINE_ALGORITHM = "LinearSearch"

COMPARISON_ALGORITHMS = [
    "LinearSearch With Preprocessing and Reduced Qubits",
    "BinSearch",
]

ALL_ALGORITHMS = [
    BASELINE_ALGORITHM,
    *COMPARISON_ALGORITHMS,
]

ALGORITHM_LABELS = {
    "LinearSearch": "LinearSearch",
    "LinearSearch With Preprocessing": "Linear + prep.",
    "LinearSearch With Reduced Qubits": "Linear + reduced",
    "LinearSearch With Preprocessing and Reduced Qubits": "Linear + all opt.",
    "BinSearch": "BinSearch + all opt.",
}


# ---------------------------------------------------------------------
# Colormaps.
# ---------------------------------------------------------------------

def viridis_like_diverging() -> LinearSegmentedColormap:
    """
    Diverging colormap visually compatible with viridis.

    Negative values use the dark purple-blue side of viridis.
    Zero is near-white.
    Positive values use the yellow side of viridis.
    """
    return LinearSegmentedColormap.from_list(
        "viridis_like_diverging",
        [
            "#440154",
            "#f7f7f7",
            "#fde725",
        ],
        N=256,
    )


BASELINE_COLORMAP = "viridis"
DIFFERENCE_COLORMAP = viridis_like_diverging()


# ---------------------------------------------------------------------
# Loading and validation.
# ---------------------------------------------------------------------

def require_columns(df: pd.DataFrame, required: set[str], name: str = "df") -> None:
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

    if df["activation_probability"].isna().any():
        bad_examples = (
            df.loc[df["activation_probability"].isna(), "problem_id"]
            .drop_duplicates()
            .head(10)
            .tolist()
        )

        raise ValueError(
            "Could not parse activation_probability from some problem IDs. "
            f"Examples: {bad_examples}"
        )

    df["embedding_found"] = df["embedding_found"].astype(float)

    return df


# ---------------------------------------------------------------------
# Aggregation logic.
# ---------------------------------------------------------------------

def aggregate_attempts_to_submissions(attempts: pd.DataFrame) -> pd.DataFrame:
    """
    First aggregation level.

    One submitted QUBO is identified by:

        problem_id, algorithm, topology, topology_size, target_cardinality,
        plus optional columns such as iteration and stage if present.

    The embedding attempts for that submitted QUBO are averaged.
    """
    optional_submission_keys = []

    for col in ["iteration", "stage"]:
        if col in attempts.columns:
            optional_submission_keys.append(col)

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
        attempts
        .groupby(submission_keys, as_index=False, dropna=False)
        .agg(
            embedding_success=("embedding_found", "mean"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
            mean_num_physical_qubits=("num_physical_qubits", "mean"),
            mean_chain_length=("mean_chain_length", "mean"),
            max_chain_length=("max_chain_length", "mean"),
            attempts_per_submission=("attempt_id", "nunique"),
            raw_attempt_rows=("attempt_id", "count"),
        )
    )

    return submission.sort_values(submission_keys).reset_index(drop=True)


def aggregate_submissions_to_algorithm_instances(submissions: pd.DataFrame) -> pd.DataFrame:
    """
    Second aggregation level.

    Multiple submitted QUBOs can belong to the same problem/algorithm pair.
    For each problem_id and algorithm:
      - embedding success is averaged over submitted QUBOs;
      - chain lengths and physical qubits are averaged over submitted QUBOs;
      - runtime is summed over submitted QUBOs;
      - num_embedding_calls is the number of submitted QUBOs.
    """
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
        submissions
        .groupby(instance_keys, as_index=False, dropna=False)
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
    """
    Load embedding_stats.csv and keep the physical/logical ratio per submitted QUBO.
    """
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

    optional_keys = []

    for col in ["iteration", "stage"]:
        if col in stats.columns:
            optional_keys.append(col)

    keys = [
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

    ratio_per_submission = (
        stats
        .groupby(keys, as_index=False, dropna=False)
        .agg(
            physical_logical_qubit_ratio=("physical_logical_qubit_ratio", "mean"),
        )
    )

    return ratio_per_submission


def add_physical_logical_ratio_to_instances(
    instances: pd.DataFrame,
    stats_path: str | Path,
) -> pd.DataFrame:
    """
    Add physical/logical qubit ratio to the algorithm-instance table.
    """
    ratio_submissions = load_embedding_stats_for_ratio(stats_path)

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
        ratio_submissions
        .groupby(instance_keys, as_index=False, dropna=False)
        .agg(
            physical_logical_qubit_ratio=("physical_logical_qubit_ratio", "mean"),
        )
    )

    merged = instances.merge(
        ratio_instances,
        on=instance_keys,
        how="left",
    )

    missing = merged["physical_logical_qubit_ratio"].isna().sum()

    if missing > 0:
        print(
            f"Warning: {missing} embedding instance rows did not receive a "
            "physical/logical qubit ratio from embedding_stats.csv."
        )

    return merged


def aggregate_instances_for_heatmaps_all_pa(instances: pd.DataFrame) -> pd.DataFrame:
    """
    Final aggregation level.

    One value per:
        algorithm, universe_size, num_test_cases

    Activation probability and repeated benchmark instances are averaged over.
    """
    heatmap_keys = [
        "algorithm",
        "universe_size",
        "num_test_cases",
    ]

    agg_dict = {
        "mean_embedding_success": ("embedding_success", "mean"),
        "mean_chain_length": ("mean_chain_length", "mean"),
        "mean_max_chain_length": ("max_chain_length", "mean"),
        "mean_num_physical_qubits": ("mean_num_physical_qubits", "mean"),
        "mean_num_embedding_calls": ("num_embedding_calls", "mean"),
        "mean_runtime_seconds": ("runtime_seconds", "mean"),
        "num_pa_values": ("activation_probability", "nunique"),
        "count": ("problem_id", "count"),
    }

    if "physical_logical_qubit_ratio" in instances.columns:
        agg_dict["mean_physical_logical_qubit_ratio"] = (
            "physical_logical_qubit_ratio",
            "mean",
        )

    agg = (
        instances
        .groupby(heatmap_keys, as_index=False, dropna=False)
        .agg(**agg_dict)
    )

    return agg.sort_values(heatmap_keys).reset_index(drop=True)


# ---------------------------------------------------------------------
# Hitman / classical comparison.
# ---------------------------------------------------------------------

def standardize_hitman_columns(hitman_df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize the Hitman runtime table.

    Expected columns:
        problem_id or problem_name
        runtime_seconds
    """
    df = hitman_df.copy()

    if "problem_name" in df.columns and "problem_id" not in df.columns:
        df = df.rename(columns={"problem_name": "problem_id"})

    require_columns(
        df,
        {
            "problem_id",
            "runtime_seconds",
        },
        name="classical comparison CSV",
    )

    return df


def collapse_hitman_runtime_per_problem(hitman_df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep one Hitman runtime per problem.
    """
    df = standardize_hitman_columns(hitman_df)

    collapsed = (
        df
        .groupby(["problem_id"], as_index=False, dropna=False)
        .agg(
            hitman_runtime_seconds=("runtime_seconds", "mean"),
            hitman_runtime_std=("runtime_seconds", "std"),
            hitman_repetitions=("runtime_seconds", "count"),
        )
    )

    return collapsed.sort_values("problem_id").reset_index(drop=True)


def add_hitman_ratio_to_instances(
    instances: pd.DataFrame,
    hitman_path: str | Path,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """
    Add embedding/Hitman runtime ratio to the algorithm-instance table.
    """
    hitman_raw = pd.read_csv(hitman_path)
    hitman_runtime = collapse_hitman_runtime_per_problem(hitman_raw)

    merged = instances.merge(
        hitman_runtime[["problem_id", "hitman_runtime_seconds"]],
        on="problem_id",
        how="inner",
    )

    if merged.empty:
        raise ValueError(
            "The merge between embedding instances and Hitman runtimes is empty. "
            "Check that embedding_attempts.problem_id matches the classical "
            "comparison problem_id/problem_name column."
        )

    missing = len(instances) - len(merged)

    if missing > 0:
        print(
            f"Warning: {missing} embedding instance rows did not find a matching "
            "Hitman runtime and were dropped from the classical comparison."
        )

    merged["embedding_over_hitman"] = (
        merged["runtime_seconds"].astype(float)
        / (merged["hitman_runtime_seconds"].astype(float) + eps)
    )

    merged["log_embedding_over_hitman"] = np.log10(
        (merged["runtime_seconds"].astype(float) + eps)
        / (merged["hitman_runtime_seconds"].astype(float) + eps)
    )

    return merged.sort_values(["algorithm", "problem_id"]).reset_index(drop=True)


def aggregate_hitman_ratio_for_heatmaps_all_pa(
    ratio_instances: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate embedding/Hitman ratio values for heatmaps.

    One value per:
        algorithm, universe_size, num_test_cases
    """
    require_columns(
        ratio_instances,
        {
            "algorithm",
            "universe_size",
            "num_test_cases",
            "activation_probability",
            "embedding_over_hitman",
            "log_embedding_over_hitman",
        },
        name="ratio_instances",
    )

    agg = (
        ratio_instances
        .groupby(
            ["algorithm", "universe_size", "num_test_cases"],
            as_index=False,
            dropna=False,
        )
        .agg(
            mean_embedding_over_hitman=("embedding_over_hitman", "mean"),
            median_embedding_over_hitman=("embedding_over_hitman", "median"),
            mean_log_embedding_over_hitman=("log_embedding_over_hitman", "mean"),
            median_log_embedding_over_hitman=("log_embedding_over_hitman", "median"),
            num_pa_values=("activation_probability", "nunique"),
            count=("problem_id", "count"),
        )
    )

    return (
        agg
        .sort_values(["algorithm", "universe_size", "num_test_cases"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------
# Table preparation for plotting.
# ---------------------------------------------------------------------

def check_algorithms_available(agg: pd.DataFrame) -> None:
    available = set(agg["algorithm"].dropna().unique().tolist())
    missing = [algorithm for algorithm in ALL_ALGORITHMS if algorithm not in available]

    if missing:
        raise ValueError(
            f"Missing required algorithms: {missing}. "
            f"Available algorithms: {sorted(available)}"
        )


def make_algorithm_table(
    agg: pd.DataFrame,
    algorithm: str,
    value_column: str,
) -> pd.DataFrame:
    return agg.loc[
        agg["algorithm"] == algorithm,
        ["universe_size", "num_test_cases", value_column],
    ].dropna(subset=["universe_size", "num_test_cases", value_column]).copy()


def make_difference_table(
    agg: pd.DataFrame,
    algorithm: str,
    baseline_algorithm: str,
    value_column: str,
    output_column: str,
    log_ratio: bool = False,
    eps: float = 1e-12,
) -> pd.DataFrame:
    baseline = make_algorithm_table(
        agg=agg,
        algorithm=baseline_algorithm,
        value_column=value_column,
    ).rename(columns={value_column: "baseline_value"})

    current = make_algorithm_table(
        agg=agg,
        algorithm=algorithm,
        value_column=value_column,
    ).rename(columns={value_column: "algorithm_value"})

    merged = current.merge(
        baseline,
        on=["universe_size", "num_test_cases"],
        how="inner",
    )

    if log_ratio:
        merged[output_column] = np.log10(
            (merged["algorithm_value"].astype(float) + eps)
            / (merged["baseline_value"].astype(float) + eps)
        )
    else:
        merged[output_column] = (
            merged["algorithm_value"].astype(float)
            - merged["baseline_value"].astype(float)
        )

    return merged[["universe_size", "num_test_cases", output_column]].copy()


# ---------------------------------------------------------------------
# Plotting utilities.
# ---------------------------------------------------------------------

def finite_values(table: pd.DataFrame, value_column: str) -> np.ndarray:
    values = table[value_column].to_numpy(dtype=float)
    return values[np.isfinite(values)]


def finite_values_from_tables(
    tables: list[pd.DataFrame],
    value_column: str,
) -> np.ndarray:
    arrays = []

    for table in tables:
        if value_column not in table.columns:
            continue

        arr = finite_values(table, value_column)

        if arr.size > 0:
            arrays.append(arr)

    if not arrays:
        return np.array([])

    return np.concatenate(arrays)


def get_numeric_extent(tables: list[pd.DataFrame]) -> tuple[float, float, float, float]:
    xs = []
    ys = []

    for table in tables:
        if table.empty:
            continue

        xs.append(table["universe_size"].to_numpy(dtype=float))
        ys.append(table["num_test_cases"].to_numpy(dtype=float))

    if not xs or not ys:
        raise ValueError("Cannot determine plot extent because all tables are empty.")

    x = np.concatenate(xs)
    y = np.concatenate(ys)

    return float(np.min(x)), float(np.max(x)), float(np.min(y)), float(np.max(y))


def expand_extent(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    margin_fraction: float = 0.03,
) -> tuple[float, float, float, float]:
    x_margin = max((xmax - xmin) * margin_fraction, 0.5)
    y_margin = max((ymax - ymin) * margin_fraction, 0.5)

    return (
        xmin - x_margin,
        xmax + x_margin,
        ymin - y_margin,
        ymax + y_margin,
    )


def interpolate_and_smooth(
    table: pd.DataFrame,
    value_column: str,
    extent: tuple[float, float, float, float],
    grid_resolution: int,
    gaussian_sigma: float,
    interpolation_method: str,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    if griddata is None or gaussian_filter is None:
        raise ImportError(
            "Smooth mode requires scipy. Install it with: pip install scipy"
        )

    xmin, xmax, ymin, ymax = extent

    x = table["universe_size"].to_numpy(dtype=float)
    y = table["num_test_cases"].to_numpy(dtype=float)
    z = table[value_column].to_numpy(dtype=float)

    grid_x, grid_y = np.meshgrid(
        np.linspace(xmin, xmax, grid_resolution),
        np.linspace(ymin, ymax, grid_resolution),
    )

    if len(table) < 3:
        zi = griddata(
            points=(x, y),
            values=z,
            xi=(grid_x, grid_y),
            method="nearest",
        )
    else:
        zi = griddata(
            points=(x, y),
            values=z,
            xi=(grid_x, grid_y),
            method=interpolation_method,
        )

        if np.isnan(zi).any():
            nearest = griddata(
                points=(x, y),
                values=z,
                xi=(grid_x, grid_y),
                method="nearest",
            )
            zi = np.where(np.isnan(zi), nearest, zi)

    if gaussian_sigma > 0:
        valid = np.isfinite(zi).astype(float)
        zi_filled = np.nan_to_num(zi, nan=0.0)

        numerator = gaussian_filter(zi_filled, sigma=gaussian_sigma)
        denominator = gaussian_filter(valid, sigma=gaussian_sigma)

        with np.errstate(divide="ignore", invalid="ignore"):
            zi = numerator / denominator

        zi[denominator <= 1e-12] = np.nan

    return zi, extent


def plot_table_on_axis(
    ax: plt.Axes,
    table: pd.DataFrame,
    value_column: str,
    extent: tuple[float, float, float, float],
    cmap: Any,
    mode: str,
    norm,
    vmin,
    vmax,
    grid_resolution: int,
    gaussian_sigma: float,
    interpolation_method: str,
    show_points: bool,
):
    if mode == "smooth":
        zi, image_extent = interpolate_and_smooth(
            table=table,
            value_column=value_column,
            extent=extent,
            grid_resolution=grid_resolution,
            gaussian_sigma=gaussian_sigma,
            interpolation_method=interpolation_method,
        )

        mappable = ax.imshow(
            zi,
            origin="lower",
            extent=image_extent,
            aspect="auto",
            cmap=cmap,
            norm=norm,
            vmin=vmin,
            vmax=vmax,
        )

        if show_points:
            ax.scatter(
                table["universe_size"],
                table["num_test_cases"],
                facecolors="none",
                edgecolors="black",
                linewidths=0.18,
                s=5,
            )

        return mappable

    if mode == "raw":
        mappable = ax.scatter(
            table["universe_size"],
            table["num_test_cases"],
            c=table[value_column],
            cmap=cmap,
            norm=norm,
            vmin=vmin,
            vmax=vmax,
            marker="s",
            s=95,
            edgecolors="none",
        )

        if show_points:
            ax.scatter(
                table["universe_size"],
                table["num_test_cases"],
                facecolors="none",
                edgecolors="black",
                linewidths=0.25,
                marker="o",
                s=18,
            )

        return mappable

    raise ValueError(f"Unknown mode: {mode!r}. Use 'raw' or 'smooth'.")


def make_centered_norm(values: np.ndarray) -> TwoSlopeNorm:
    if values.size == 0:
        max_abs = 1.0
    else:
        max_abs = max(abs(float(np.nanmin(values))), abs(float(np.nanmax(values))))

        if max_abs == 0:
            max_abs = 1.0

    return TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)


def build_metric_tables(
    agg: pd.DataFrame,
    value_column: str,
    output_column: str,
    log_ratio: bool,
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    baseline_table = make_algorithm_table(
        agg=agg,
        algorithm=BASELINE_ALGORITHM,
        value_column=value_column,
    )

    diff_tables = [
        make_difference_table(
            agg=agg,
            algorithm=algorithm,
            baseline_algorithm=BASELINE_ALGORITHM,
            value_column=value_column,
            output_column=output_column,
            log_ratio=log_ratio,
        )
        for algorithm in COMPARISON_ALGORITHMS
    ]

    return baseline_table, diff_tables


def make_baseline_norm(
    baseline_table: pd.DataFrame,
    value_column: str,
    baseline_vmin: float | None,
    baseline_vmax: float | None,
    baseline_center_zero: bool,
):
    baseline_values = finite_values(baseline_table, value_column)

    if baseline_values.size == 0:
        raise ValueError(f"No finite baseline values for column {value_column!r}.")

    if baseline_center_zero:
        return make_centered_norm(baseline_values), None, None

    if baseline_vmin is None:
        baseline_vmin = float(np.nanmin(baseline_values))

    if baseline_vmax is None:
        baseline_vmax = float(np.nanmax(baseline_values))

    return None, baseline_vmin, baseline_vmax


def make_difference_norm(
    diff_tables: list[pd.DataFrame],
    value_column: str,
    diff_center_zero: bool = True,
):
    diff_values = finite_values_from_tables(diff_tables, value_column)

    if diff_values.size == 0:
        raise ValueError(f"No finite difference values for column {value_column!r}.")

    if diff_center_zero:
        return make_centered_norm(diff_values), None, None

    return None, float(np.nanmin(diff_values)), float(np.nanmax(diff_values))


def add_in_panel_label(
    ax: plt.Axes,
    text: str,
    loc: str = "upper left",
    fontsize: int = 8,
) -> None:
    if loc == "upper left":
        x, y = 0.03, 0.95
        ha, va = "left", "top"
    elif loc == "upper right":
        x, y = 0.97, 0.95
        ha, va = "right", "top"
    else:
        x, y = 0.03, 0.95
        ha, va = "left", "top"

    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=fontsize,
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.72,
        },
    )


def plot_rq3_composite_figure(
    metric_specs: list[dict],
    output_path: str | Path,
    mode: str,
    grid_resolution: int = 50,
    gaussian_sigma: float = 2.0,
    interpolation_method: str = "linear",
    show_points: bool = True,
    y_label: str = "Number of test cases",
    output_format: str | None = None,
) -> None:
    """
    Create one RQ3 composite figure.

    Each row contains:

        LinearSearch absolute heatmap | baseline colorbar |
        Linear + all opt. comparison | BinSearch + all opt. comparison |
        comparison colorbar
    """

    prepared_rows = []
    all_tables_for_extent = []

    for spec in metric_specs:
        baseline_table, diff_tables = build_metric_tables(
            agg=spec["agg"],
            value_column=spec["value_column"],
            output_column=spec["output_column"],
            log_ratio=spec["log_ratio"],
        )

        baseline_norm, baseline_vmin_to_use, baseline_vmax_to_use = make_baseline_norm(
            baseline_table=baseline_table,
            value_column=spec["value_column"],
            baseline_vmin=spec["baseline_vmin"],
            baseline_vmax=spec["baseline_vmax"],
            baseline_center_zero=spec["baseline_center_zero"],
        )

        diff_norm, diff_vmin_to_use, diff_vmax_to_use = make_difference_norm(
            diff_tables=diff_tables,
            value_column=spec["output_column"],
            diff_center_zero=True,
        )

        prepared_rows.append(
            {
                "spec": spec,
                "baseline_table": baseline_table,
                "diff_tables": diff_tables,
                "baseline_norm": baseline_norm,
                "baseline_vmin": baseline_vmin_to_use,
                "baseline_vmax": baseline_vmax_to_use,
                "diff_norm": diff_norm,
                "diff_vmin": diff_vmin_to_use,
                "diff_vmax": diff_vmax_to_use,
            }
        )

        all_tables_for_extent.append(baseline_table)
        all_tables_for_extent.extend(diff_tables)

    if not prepared_rows:
        print(f"Skipping empty composite figure: {output_path}")
        return

    extent = expand_extent(*get_numeric_extent(all_tables_for_extent))

    nrows = len(prepared_rows)

    # Same visual format as RQ1/RQ2.
    # Height scales with the number of metric rows.
    fig_height = 2.08 * nrows + 0.60
    fig = plt.figure(figsize=(9.20, fig_height))

    gs = fig.add_gridspec(
        nrows=nrows,
        ncols=9,
        width_ratios=[
            1.0,    # 0 baseline heatmap
            0.045,  # 1 baseline colorbar
            0.18,   # 2 spacer
            1.0,    # 3 diff heatmap 1
            0.10,   # 4 spacer
            1.0,    # 5 diff heatmap 2
            0.045,  # 6 diff colorbar
            0.035,  # 7 right spacer
            0.015,  # 8 tiny right padding
        ],
        height_ratios=[1.0] * nrows,
        left=0.085,
        right=0.985,
        bottom=0.065,
        top=0.980,
        wspace=0.055,
        hspace=0.125,
    )

    for row_idx, row in enumerate(prepared_rows):
        spec = row["spec"]

        ax_baseline = fig.add_subplot(gs[row_idx, 0])
        cax_baseline = fig.add_subplot(gs[row_idx, 1])
        ax_diff_1 = fig.add_subplot(gs[row_idx, 3], sharex=ax_baseline, sharey=ax_baseline)
        ax_diff_2 = fig.add_subplot(gs[row_idx, 5], sharex=ax_baseline, sharey=ax_baseline)
        cax_diff = fig.add_subplot(gs[row_idx, 6])

        row_axes = [ax_baseline, ax_diff_1, ax_diff_2]

        baseline_mappable = plot_table_on_axis(
            ax=ax_baseline,
            table=row["baseline_table"],
            value_column=spec["value_column"],
            extent=extent,
            cmap=spec["baseline_cmap"],
            mode=mode,
            norm=row["baseline_norm"],
            vmin=row["baseline_vmin"],
            vmax=row["baseline_vmax"],
            grid_resolution=grid_resolution,
            gaussian_sigma=gaussian_sigma,
            interpolation_method=interpolation_method,
            show_points=show_points,
        )

        diff_mappable_1 = plot_table_on_axis(
            ax=ax_diff_1,
            table=row["diff_tables"][0],
            value_column=spec["output_column"],
            extent=extent,
            cmap=spec["diff_cmap"],
            mode=mode,
            norm=row["diff_norm"],
            vmin=row["diff_vmin"],
            vmax=row["diff_vmax"],
            grid_resolution=grid_resolution,
            gaussian_sigma=gaussian_sigma,
            interpolation_method=interpolation_method,
            show_points=show_points,
        )

        diff_mappable_2 = plot_table_on_axis(
            ax=ax_diff_2,
            table=row["diff_tables"][1],
            value_column=spec["output_column"],
            extent=extent,
            cmap=spec["diff_cmap"],
            mode=mode,
            norm=row["diff_norm"],
            vmin=row["diff_vmin"],
            vmax=row["diff_vmax"],
            grid_resolution=grid_resolution,
            gaussian_sigma=gaussian_sigma,
            interpolation_method=interpolation_method,
            show_points=show_points,
        )

        baseline_cbar = fig.colorbar(
            baseline_mappable,
            cax=cax_baseline,
            orientation="vertical",
        )
        baseline_cbar.set_label(spec["baseline_colorbar_label"], fontsize=6.5, labelpad=3)
        baseline_cbar.ax.tick_params(labelsize=5.8, pad=1, length=2)

        diff_cbar = fig.colorbar(
            diff_mappable_2,
            cax=cax_diff,
            orientation="vertical",
        )
        diff_cbar.set_label(spec["diff_colorbar_label"], fontsize=6.5, labelpad=3)
        diff_cbar.ax.tick_params(labelsize=5.8, pad=1, length=2)

        # Row label on the left.
        ax_baseline.text(
            -0.34,
            0.50,
            spec["row_label"],
            transform=ax_baseline.transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=7.7,
            fontweight="bold",
        )

        # Algorithm labels only on the first row.
        if row_idx == 0:
            add_in_panel_label(
                ax_baseline,
                ALGORITHM_LABELS[BASELINE_ALGORITHM],
                fontsize=7.0,
            )
            add_in_panel_label(
                ax_diff_1,
                ALGORITHM_LABELS[COMPARISON_ALGORITHMS[0]],
                fontsize=7.0,
            )
            add_in_panel_label(
                ax_diff_2,
                ALGORITHM_LABELS[COMPARISON_ALGORITHMS[1]],
                fontsize=7.0,
            )

        for ax in row_axes:
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            ax.grid(False)
            ax.tick_params(axis="both", which="major", labelsize=6.5, length=2.3)

        ax_baseline.set_ylabel(y_label, fontsize=7.0)
        ax_diff_1.set_ylabel("")
        ax_diff_2.set_ylabel("")

        ax_diff_1.tick_params(labelleft=False)
        ax_diff_2.tick_params(labelleft=False)

        # Shared horizontal axis:
        # show x labels only on the bottom row.
        if row_idx < nrows - 1:
            for ax in row_axes:
                ax.tick_params(labelbottom=False)
                ax.set_xlabel("")
        else:
            for ax in row_axes:
                ax.set_xlabel("Universe size", fontsize=7.2)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_format is None:
        output_format = output_path.suffix.lstrip(".") or "pdf"

    fig.savefig(output_path, dpi=300, bbox_inches="tight", format=output_format)
    plt.close(fig)


# ---------------------------------------------------------------------
# Main run.
# ---------------------------------------------------------------------

def run(
    attempts_path: str | Path,
    output_dir: str | Path,
    mode: str,
    grid_resolution: int,
    gaussian_sigma: float,
    interpolation_method: str,
    show_points: bool,
    y_label: str,
    classical_comparison_path: str | Path | None = None,
    stats_path: str | Path | None = None,
    output_format: str = "pdf",
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    attempts = load_embedding_attempts(attempts_path)
    submissions = aggregate_attempts_to_submissions(attempts)
    instances = aggregate_submissions_to_algorithm_instances(submissions)

    if stats_path is not None:
        instances = add_physical_logical_ratio_to_instances(
            instances=instances,
            stats_path=stats_path,
        )

    agg = aggregate_instances_for_heatmaps_all_pa(instances)

    check_algorithms_available(agg)

    suffix = f"all_pa_{mode}"
    baseline_cmap = BASELINE_COLORMAP
    diff_cmap = DIFFERENCE_COLORMAP

    # ------------------------------------------------------------
    # Figure 1:
    # Chain length and physical overhead.
    # ------------------------------------------------------------
    chain_overhead_metric_specs = [
        {
            "agg": agg,
            "row_label": "Mean chain length",
            "value_column": "mean_chain_length",
            "output_column": "diff_mean_chain_length",
            "baseline_colorbar_label": "Mean chain length",
            "diff_colorbar_label": r"$\Delta$ mean chain length",
            "log_ratio": False,
            "baseline_vmin": None,
            "baseline_vmax": None,
            "baseline_center_zero": False,
            "baseline_cmap": baseline_cmap,
            "diff_cmap": diff_cmap,
        },
        {
            "agg": agg,
            "row_label": "Max chain length",
            "value_column": "mean_max_chain_length",
            "output_column": "diff_max_chain_length",
            "baseline_colorbar_label": "Max chain length",
            "diff_colorbar_label": r"$\Delta$ max chain length",
            "log_ratio": False,
            "baseline_vmin": None,
            "baseline_vmax": None,
            "baseline_center_zero": False,
            "baseline_cmap": baseline_cmap,
            "diff_cmap": diff_cmap,
        },
        {
            "agg": agg,
            "row_label": "Physical qubits",
            "value_column": "mean_num_physical_qubits",
            "output_column": "diff_num_physical_qubits",
            "baseline_colorbar_label": "Physical qubits",
            "diff_colorbar_label": r"$\Delta$ physical qubits",
            "log_ratio": False,
            "baseline_vmin": None,
            "baseline_vmax": None,
            "baseline_center_zero": False,
            "baseline_cmap": baseline_cmap,
            "diff_cmap": diff_cmap,
        },
    ]

    if "mean_physical_logical_qubit_ratio" in agg.columns:
        chain_overhead_metric_specs.append(
            {
                "agg": agg,
                "row_label": "Physical/logical ratio",
                "value_column": "mean_physical_logical_qubit_ratio",
                "output_column": "diff_physical_logical_qubit_ratio",
                "baseline_colorbar_label": "Physical/logical ratio",
                "diff_colorbar_label": r"$\Delta$ physical/logical ratio",
                "log_ratio": False,
                "baseline_vmin": None,
                "baseline_vmax": None,
                "baseline_center_zero": False,
                "baseline_cmap": baseline_cmap,
                "diff_cmap": diff_cmap,
            }
        )
    else:
        print(
            "Physical/logical qubit ratio will not be included in the chain/overhead "
            "figure because no stats CSV was provided."
        )

    plot_rq3_composite_figure(
        metric_specs=chain_overhead_metric_specs,
        output_path=output_dir / f"rq3_chain_overhead_composite_{suffix}.{output_format}",
        mode=mode,
        grid_resolution=grid_resolution,
        gaussian_sigma=gaussian_sigma,
        interpolation_method=interpolation_method,
        show_points=show_points,
        y_label=y_label,
        output_format=output_format,
    )

    # ------------------------------------------------------------
    # Figure 2:
    # All other embedding metrics.
    # ------------------------------------------------------------
    other_metric_specs = [
        {
            "agg": agg,
            "row_label": "Embedding success",
            "value_column": "mean_embedding_success",
            "output_column": "diff_embedding_success",
            "baseline_colorbar_label": "Embedding success",
            "diff_colorbar_label": r"$\Delta$ embedding success",
            "log_ratio": False,
            "baseline_vmin": 0.0,
            "baseline_vmax": 1.0,
            "baseline_center_zero": False,
            "baseline_cmap": baseline_cmap,
            "diff_cmap": diff_cmap,
        },
        {
            "agg": agg,
            "row_label": "Embedding calls",
            "value_column": "mean_num_embedding_calls",
            "output_column": "diff_num_embedding_calls",
            "baseline_colorbar_label": "Embedding calls",
            "diff_colorbar_label": r"$\Delta$ embedding calls",
            "log_ratio": False,
            "baseline_vmin": None,
            "baseline_vmax": None,
            "baseline_center_zero": False,
            "baseline_cmap": baseline_cmap,
            "diff_cmap": diff_cmap,
        },
        {
            "agg": agg,
            "row_label": "Runtime",
            "value_column": "mean_runtime_seconds",
            "output_column": "runtime_log_ratio",
            "baseline_colorbar_label": "Runtime [s]",
            "diff_colorbar_label": r"$\log_{10}(T_{\mathrm{method}}/T_{\mathrm{baseline}})$",
            "log_ratio": True,
            "baseline_vmin": None,
            "baseline_vmax": None,
            "baseline_center_zero": False,
            "baseline_cmap": baseline_cmap,
            "diff_cmap": diff_cmap,
        },
    ]

    # Save the intermediate CSVs before optional Hitman processing.
    submissions.to_csv(
        output_dir / "embedding_submissions_averaged_over_attempts.csv",
        index=False,
    )

    instances.to_csv(
        output_dir / "embedding_algorithm_instances.csv",
        index=False,
    )

    agg.to_csv(
        output_dir / "embedding_heatmap_values_all_pa.csv",
        index=False,
    )

    if classical_comparison_path is not None:
        ratio_instances = add_hitman_ratio_to_instances(
            instances=instances,
            hitman_path=classical_comparison_path,
        )

        ratio_agg = aggregate_hitman_ratio_for_heatmaps_all_pa(
            ratio_instances=ratio_instances,
        )

        ratio_instances.to_csv(
            output_dir / "embedding_hitman_ratio_instances.csv",
            index=False,
        )

        ratio_agg.to_csv(
            output_dir / "embedding_hitman_ratio_heatmap_values_all_pa.csv",
            index=False,
        )

        other_metric_specs.append(
            {
                "agg": ratio_agg,
                "row_label": "Emb./Hitman runtime",
                "value_column": "mean_log_embedding_over_hitman",
                "output_column": "diff_log_embedding_over_hitman",
                "baseline_colorbar_label": r"$\log_{10}(T_{\mathrm{emb}}/T_{\mathrm{Hitman}})$",
                "diff_colorbar_label": r"$\Delta \log_{10}(T_{\mathrm{emb}}/T_{\mathrm{Hitman}})$",
                "log_ratio": False,
                "baseline_vmin": None,
                "baseline_vmax": None,
                "baseline_center_zero": True,
                "baseline_cmap": diff_cmap,
                "diff_cmap": diff_cmap,
            }
        )

    plot_rq3_composite_figure(
        metric_specs=other_metric_specs,
        output_path=output_dir / f"rq3_other_metrics_composite_{suffix}.{output_format}",
        mode=mode,
        grid_resolution=grid_resolution,
        gaussian_sigma=gaussian_sigma,
        interpolation_method=interpolation_method,
        show_points=show_points,
        y_label=y_label,
        output_format=output_format,
    )

    print(f"Saved RQ3 composite heatmaps to: {output_dir}")


# ---------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create RQ3 composite heatmaps. The script first averages the embedding "
            "attempts for each submitted QUBO, then sums runtime over all submitted "
            "QUBOs belonging to the same algorithm/problem instance, and finally "
            "averages over activation probabilities for the heatmaps."
        )
    )

    parser.add_argument(
        "--attempts",
        required=True,
        type=Path,
        help="Path to embedding_attempts.csv.",
    )

    parser.add_argument(
        "--stats",
        type=Path,
        default=None,
        help=(
            "Optional path to embedding_stats.csv. If provided, the script also "
            "includes the physical/logical qubit ratio in the chain/overhead figure."
        ),
    )

    parser.add_argument(
        "--classical-comparison",
        type=Path,
        default=None,
        help=(
            "Optional path to Hitman/classical runtime CSV. The file must contain "
            "runtime_seconds and either problem_id or problem_name. If provided, "
            "the script also includes an embedding-over-Hitman runtime ratio row."
        ),
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where plots and aggregated CSV files will be saved.",
    )

    parser.add_argument(
        "--mode",
        choices=["raw", "smooth"],
        default="smooth",
        help=(
            "raw: plot actual aggregated data points with numerical spacing. "
            "smooth: interpolate to a dense grid and apply Gaussian smoothing."
        ),
    )

    parser.add_argument(
        "--grid-resolution",
        type=int,
        default=50,
        help="Dense grid resolution used in smooth mode.",
    )

    parser.add_argument(
        "--gaussian-sigma",
        type=float,
        default=2.0,
        help="Gaussian smoothing sigma used in smooth mode.",
    )

    parser.add_argument(
        "--interpolation-method",
        choices=["linear", "nearest", "cubic"],
        default="linear",
        help="Interpolation method used before smoothing.",
    )

    parser.add_argument(
        "--hide-points",
        action="store_true",
        help="Do not overlay original aggregated data-point locations.",
    )

    parser.add_argument(
        "--y-label",
        type=str,
        default="Number of test cases",
        help=(
            "Y-axis label. Use 'Effective number of test cases' if "
            "num_test_cases is after preprocessing."
        ),
    )

    parser.add_argument(
        "--format",
        choices=["pdf", "png", "svg"],
        default="pdf",
        help="Output format for the composite figures.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run(
        attempts_path=args.attempts,
        output_dir=args.output_dir,
        mode=args.mode,
        grid_resolution=args.grid_resolution,
        gaussian_sigma=args.gaussian_sigma,
        interpolation_method=args.interpolation_method,
        show_points=not args.hide_points,
        y_label=args.y_label,
        classical_comparison_path=args.classical_comparison,
        stats_path=args.stats,
        output_format=args.format,
    )