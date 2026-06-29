from __future__ import annotations

import argparse
import re
from pathlib import Path

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


BASELINE_ALGORITHM = "CardinalitySweep"

COMPARISON_ALGORITHMS = [
    "RecursivePUBO",
    "RecursivePUBO With Preprocessing and Adaptive Qubits",
]

ALL_ALGORITHMS = [
    BASELINE_ALGORITHM,
    *COMPARISON_ALGORITHMS,
]

ALGORITHM_LABELS = {
    "CardinalitySweep": "Cardinality Sweep",
    "RecursivePUBO": "Recursive PUBO",
    "RecursivePUBO With Preprocessing and Adaptive Qubits": "Recursive PUBO + all opt.",
}


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


def load_and_prepare_global_stats(path: str | Path) -> pd.DataFrame:
    path = Path(path)
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

    return df


def aggregate_for_heatmaps_all_pa(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate one value per:

        algorithm, universe_size, num_test_cases

    averaging over all activation probabilities and repeated runs/seeds.
    """
    require_columns(
        df,
        {
            "algorithm",
            "universe_size",
            "num_test_cases",
            "activation_probability",
            "global_found_ratio",
            "num_qubits",
            "num_couplers",
            "runtime_seconds",
        },
        name="global_stats.csv",
    )

    agg = (
        df
        .groupby(
            ["algorithm", "universe_size", "num_test_cases"],
            as_index=False,
            dropna=False,
        )
        .agg(
            mean_global_found_ratio=("global_found_ratio", "mean"),
            mean_num_qubits=("num_qubits", "mean"),
            mean_num_couplers=("num_couplers", "mean"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
            num_pa_values=("activation_probability", "nunique"),
            count=("global_found_ratio", "count"),
        )
    )

    return (
        agg
        .sort_values(["algorithm", "universe_size", "num_test_cases"])
        .reset_index(drop=True)
    )


def check_algorithms_available(agg: pd.DataFrame) -> None:
    available = set(agg["algorithm"].dropna().unique().tolist())
    missing = [alg for alg in ALL_ALGORITHMS if alg not in available]

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
    cmap,
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


def plot_rq1_composite_figure(
    agg: pd.DataFrame,
    output_path: str | Path,
    mode: str,
    grid_resolution: int = 250,
    gaussian_sigma: float = 2.0,
    interpolation_method: str = "linear",
    show_points: bool = True,
    y_label: str = "Number of test cases",
    output_format: str | None = None,
) -> None:
    """
    Create one single RQ1 figure with four rows:

        1. Global found ratio
        2. Runtime
        3. Logical variables
        4. Logical couplers

    Each row contains:

        baseline absolute heatmap | baseline colorbar |
        RecursivePUBO comparison | RecursivePUBO + all opt. comparison |
        comparison colorbar

    For success metrics, comparison = method - baseline.
    For runtime, comparison = log10(method / baseline).
    For logical variables and logical couplers, comparison = method - baseline.
    """

    metric_specs = [
        {
            "row_label": "Global found ratio",
            "value_column": "mean_global_found_ratio",
            "output_column": "diff_global_found_ratio",
            "baseline_colorbar_label": "Global found ratio",
            "diff_colorbar_label": r"$\Delta$ global found ratio",
            "log_ratio": False,
            "baseline_vmin": 0.0,
            "baseline_vmax": 1.0,
            "baseline_center_zero": False,
        },
        {
            "row_label": "Runtime",
            "value_column": "mean_runtime_seconds",
            "output_column": "runtime_log_ratio",
            "baseline_colorbar_label": "Runtime [s]",
            "diff_colorbar_label": r"$\log_{10}(T_{\mathrm{method}}/T_{\mathrm{baseline}})$",
            "log_ratio": True,
            "baseline_vmin": None,
            "baseline_vmax": None,
            "baseline_center_zero": False,
        },
        {
            "row_label": "Logical variables",
            "value_column": "mean_num_qubits",
            "output_column": "diff_num_qubits",
            "baseline_colorbar_label": "Logical variables",
            "diff_colorbar_label": r"$\Delta$ logical variables",
            "log_ratio": False,
            "baseline_vmin": None,
            "baseline_vmax": None,
            "baseline_center_zero": False,
        },
        {
            "row_label": "Logical couplers",
            "value_column": "mean_num_couplers",
            "output_column": "diff_num_couplers",
            "baseline_colorbar_label": "Logical couplers",
            "diff_colorbar_label": r"$\Delta$ logical couplers",
            "log_ratio": False,
            "baseline_vmin": None,
            "baseline_vmax": None,
            "baseline_center_zero": False,
        },
    ]

    prepared_rows = []

    all_tables_for_extent = []

    for spec in metric_specs:
        baseline_table, diff_tables = build_metric_tables(
            agg=agg,
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

    extent = expand_extent(*get_numeric_extent(all_tables_for_extent))

    # Page-sized figure.
    # This is suitable for a full-page figure in a paper.
    fig = plt.figure(figsize=(9.20, 9.40))

    gs = fig.add_gridspec(
        nrows=4,
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
        height_ratios=[1.0, 1.0, 1.0, 1.0],
        left=0.085,
        right=0.985,
        bottom=0.070,
        top=0.975,
        wspace=0.055,
        hspace=0.135,
    )

    axes_grid: list[list[plt.Axes]] = []

    for row_idx, row in enumerate(prepared_rows):
        spec = row["spec"]

        ax_baseline = fig.add_subplot(gs[row_idx, 0])
        cax_baseline = fig.add_subplot(gs[row_idx, 1])
        ax_diff_1 = fig.add_subplot(gs[row_idx, 3], sharex=ax_baseline, sharey=ax_baseline)
        ax_diff_2 = fig.add_subplot(gs[row_idx, 5], sharex=ax_baseline, sharey=ax_baseline)
        cax_diff = fig.add_subplot(gs[row_idx, 6])

        row_axes = [ax_baseline, ax_diff_1, ax_diff_2]
        axes_grid.append(row_axes)

        baseline_mappable = plot_table_on_axis(
            ax=ax_baseline,
            table=row["baseline_table"],
            value_column=spec["value_column"],
            extent=extent,
            cmap=BASELINE_COLORMAP,
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
            cmap=DIFFERENCE_COLORMAP,
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
            cmap=DIFFERENCE_COLORMAP,
            mode=mode,
            norm=row["diff_norm"],
            vmin=row["diff_vmin"],
            vmax=row["diff_vmax"],
            grid_resolution=grid_resolution,
            gaussian_sigma=gaussian_sigma,
            interpolation_method=interpolation_method,
            show_points=show_points,
        )

        baseline_cbar = fig.colorbar(baseline_mappable, cax=cax_baseline)
        baseline_cbar.set_label(spec["baseline_colorbar_label"], fontsize=6.8)
        baseline_cbar.ax.tick_params(labelsize=6.2, length=2)

        diff_cbar = fig.colorbar(diff_mappable_2, cax=cax_diff)
        diff_cbar.set_label(spec["diff_colorbar_label"], fontsize=6.8)
        diff_cbar.ax.tick_params(labelsize=6.2, length=2)

        # Row label on the left.
        ax_baseline.text(
            -0.34,
            0.50,
            spec["row_label"],
            transform=ax_baseline.transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=8.2,
            fontweight="bold",
        )

        # Algorithm labels only on the first row.
        if row_idx == 0:
            add_in_panel_label(
                ax_baseline,
                ALGORITHM_LABELS[BASELINE_ALGORITHM],
                fontsize=7.3,
            )
            add_in_panel_label(
                ax_diff_1,
                ALGORITHM_LABELS[COMPARISON_ALGORITHMS[0]],
                fontsize=7.3,
            )
            add_in_panel_label(
                ax_diff_2,
                ALGORITHM_LABELS[COMPARISON_ALGORITHMS[1]],
                fontsize=7.3,
            )

        for ax in row_axes:
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            ax.grid(False)
            ax.tick_params(axis="both", which="major", labelsize=6.8, length=2.5)

        # Only the first panel in each row gets the y-axis label/ticks.
        ax_baseline.set_ylabel(y_label, fontsize=7.2)
        ax_diff_1.set_ylabel("")
        ax_diff_2.set_ylabel("")

        ax_diff_1.tick_params(labelleft=False)
        ax_diff_2.tick_params(labelleft=False)

        # Hide x tick labels except on the bottom row.
        if row_idx < len(prepared_rows) - 1:
            for ax in row_axes:
                ax.tick_params(labelbottom=False)
                ax.set_xlabel("")
        else:
            for ax in row_axes:
                ax.set_xlabel("Universe size", fontsize=7.5)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_format is None:
        output_format = output_path.suffix.lstrip(".") or "pdf"

    fig.savefig(output_path, dpi=300, bbox_inches="tight", format=output_format)
    plt.close(fig)


def run(
    input_path: str | Path,
    output_dir: str | Path,
    mode: str,
    grid_resolution: int,
    gaussian_sigma: float,
    interpolation_method: str,
    show_points: bool,
    y_label: str,
    output_format: str,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_prepare_global_stats(input_path)
    agg = aggregate_for_heatmaps_all_pa(df)

    check_algorithms_available(agg)

    suffix = f"all_pa_{mode}"

    output_path = output_dir / f"rq1_composite_{suffix}.{output_format}"

    plot_rq1_composite_figure(
        agg=agg,
        output_path=output_path,
        mode=mode,
        grid_resolution=grid_resolution,
        gaussian_sigma=gaussian_sigma,
        interpolation_method=interpolation_method,
        show_points=show_points,
        y_label=y_label,
        output_format=output_format,
    )

    agg.to_csv(output_dir / "rq1_aggregated_heatmap_values_all_pa.csv", index=False)

    print(f"Saved RQ1 composite figure to: {output_path}")
    print(f"Saved aggregated values to: {output_dir / 'rq1_aggregated_heatmap_values_all_pa.csv'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create one single RQ1 composite heatmap figure averaged over all "
            "activation probabilities. The figure has four rows: global found ratio, "
            "runtime, logical variables, and logical couplers."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to global_stats.csv.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where the composite plot will be saved.",
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
        help="Output format for the composite figure.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run(
        input_path=args.input,
        output_dir=args.output_dir,
        mode=args.mode,
        grid_resolution=args.grid_resolution,
        gaussian_sigma=args.gaussian_sigma,
        interpolation_method=args.interpolation_method,
        show_points=not args.hide_points,
        y_label=args.y_label,
        output_format=args.format,
    )