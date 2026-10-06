from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

GREEN = "\033[92m"
RESET = "\033[0m"


def pass_text() -> str:
    if sys.stdout.isatty():
        return f"{GREEN}PASS{RESET}"
    return "PASS"


DEFAULT_RTOL = 1e-9
DEFAULT_ATOL = 1e-12


def load_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")

    return pd.read_csv(path)


def normalized_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize row order and column order before comparison.

    Column names are sorted alphabetically. Rows are sorted by all columns
    using a stable sort where possible.

    This makes the comparison insensitive to harmless CSV row ordering.
    """
    df = df.copy()

    columns = sorted(df.columns.tolist())
    df = df[columns]

    try:
        df = df.sort_values(
            by=columns,
            kind="mergesort",
            na_position="last",
        )
    except TypeError:
        # Some mixed-type columns may not be directly sortable.
        # In that case, preserve the original row order.
        pass

    return df.reset_index(drop=True)


def compare_dataframes(
    reference: pd.DataFrame,
    reproduced: pd.DataFrame,
    rtol: float,
    atol: float,
) -> tuple[bool, list[str]]:
    errors: list[str] = []

    reference = normalized_dataframe(reference)
    reproduced = normalized_dataframe(reproduced)

    reference_columns = set(reference.columns)
    reproduced_columns = set(reproduced.columns)

    if reference_columns != reproduced_columns:
        missing = sorted(reference_columns - reproduced_columns)
        extra = sorted(reproduced_columns - reference_columns)

        if missing:
            errors.append(
                f"Missing columns in reproduced CSV: {missing}"
            )

        if extra:
            errors.append(
                f"Unexpected columns in reproduced CSV: {extra}"
            )

        return False, errors

    if len(reference) != len(reproduced):
        errors.append(
            "Different number of rows: "
            f"reference={len(reference)}, reproduced={len(reproduced)}"
        )
        return False, errors

    for column in reference.columns:
        ref_col = reference[column]
        new_col = reproduced[column]

        ref_numeric = pd.to_numeric(ref_col, errors="coerce")
        new_numeric = pd.to_numeric(new_col, errors="coerce")

        # Treat as numeric only when all non-null original values
        # could be converted to numbers.
        ref_numeric_valid = (
            ref_col.isna() | ref_numeric.notna()
        ).all()

        new_numeric_valid = (
            new_col.isna() | new_numeric.notna()
        ).all()

        if ref_numeric_valid and new_numeric_valid:
            ref_values = ref_numeric.to_numpy(dtype=float)
            new_values = new_numeric.to_numpy(dtype=float)

            equal = np.isclose(
                ref_values,
                new_values,
                rtol=rtol,
                atol=atol,
                equal_nan=True,
            )

            if not np.all(equal):
                bad = np.where(~equal)[0]

                examples = []
                for idx in bad[:5]:
                    examples.append(
                        f"row {idx}: "
                        f"reference={ref_values[idx]!r}, "
                        f"reproduced={new_values[idx]!r}"
                    )

                errors.append(
                    f"Numeric mismatch in column {column!r} "
                    f"({len(bad)} rows differ). "
                    + "; ".join(examples)
                )

        else:
            ref_values = ref_col.fillna("<NA>").astype(str)
            new_values = new_col.fillna("<NA>").astype(str)

            equal = ref_values == new_values

            if not equal.all():
                bad = np.where(~equal.to_numpy())[0]

                examples = []
                for idx in bad[:5]:
                    examples.append(
                        f"row {idx}: "
                        f"reference={ref_values.iloc[idx]!r}, "
                        f"reproduced={new_values.iloc[idx]!r}"
                    )

                errors.append(
                    f"String mismatch in column {column!r} "
                    f"({len(bad)} rows differ). "
                    + "; ".join(examples)
                )

    return len(errors) == 0, errors


def compare_csv(
    name: str,
    reference_path: Path,
    reproduced_path: Path,
    rtol: float,
    atol: float,
) -> bool:
    print(f"Checking {name} ... ", end="", flush=True)

    try:
        reference = load_csv(reference_path)
        reproduced = load_csv(reproduced_path)
    except Exception as exc:
        print("FAIL")
        print(f"  {exc}")
        return False

    ok, errors = compare_dataframes(
        reference=reference,
        reproduced=reproduced,
        rtol=rtol,
        atol=atol,
    )

    if ok:
        print(pass_text())
        return True

    print("FAIL")

    for error in errors:
        print(f"  - {error}")

    print(f"  Reference:  {reference_path}")
    print(f"  Reproduced: {reproduced_path}")

    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare aggregated CSV files regenerated by the reproduction "
            "pipeline with the frozen aggregated CSV files used in the paper."
        )
    )

    parser.add_argument(
        "--reference-dir",
        required=True,
        type=Path,
        help="Directory containing the frozen paper aggregated CSVs.",
    )

    parser.add_argument(
        "--reproduction-dir",
        required=True,
        type=Path,
        help="Root reproduction directory containing regenerated outputs.",
    )

    parser.add_argument(
        "--rtol",
        type=float,
        default=DEFAULT_RTOL,
        help=f"Relative tolerance for numeric comparisons. Default: {DEFAULT_RTOL}",
    )

    parser.add_argument(
        "--atol",
        type=float,
        default=DEFAULT_ATOL,
        help=f"Absolute tolerance for numeric comparisons. Default: {DEFAULT_ATOL}",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ref = args.reference_dir
    out = args.reproduction_dir

    checks = [
        # ------------------------------------------------------------
        # Paper summary tables
        # ------------------------------------------------------------

        (
            "RQ1 summary table",
            ref / "tables" / "table_rq1_summary.csv",
            out / "tables" / "table_rq1_summary.csv",
        ),
        (
            "RQ2 summary table",
            ref / "tables" / "table_rq2_summary.csv",
            out / "tables" / "table_rq2_summary.csv",
        ),
        (
            "RQ3 summary table",
            ref / "tables" / "table_rq3_summary.csv",
            out / "tables" / "table_rq3_summary.csv",
        ),
        (
            "RQ1 aggregated heatmap data",
            ref / "rq1_aggregated_heatmap_values_all_pa.csv",
            out
            / "figures"
            / "rq1"
            / "rq1_aggregated_heatmap_values_all_pa.csv",
        ),
        (
            "RQ2 aggregated heatmap data",
            ref / "rq2_aggregated_heatmap_values_all_pa.csv",
            out
            / "figures"
            / "rq2"
            / "rq2_aggregated_heatmap_values_all_pa.csv",
        ),

        # Zephyr
        (
            "RQ3 Zephyr submissions",
            ref
            / "zephyr"
            / "embedding_submissions_averaged_over_attempts.csv",
            out
            / "figures"
            / "rq3_zephyr"
            / "embedding_submissions_averaged_over_attempts.csv",
        ),
        (
            "RQ3 Zephyr algorithm instances",
            ref
            / "zephyr"
            / "embedding_algorithm_instances.csv",
            out
            / "figures"
            / "rq3_zephyr"
            / "embedding_algorithm_instances.csv",
        ),
        (
            "RQ3 Zephyr heatmap values",
            ref
            / "zephyr"
            / "embedding_heatmap_values_all_pa.csv",
            out
            / "figures"
            / "rq3_zephyr"
            / "embedding_heatmap_values_all_pa.csv",
        ),
        (
            "RQ3 Zephyr Hitman-ratio instances",
            ref
            / "zephyr"
            / "embedding_hitman_ratio_instances.csv",
            out
            / "figures"
            / "rq3_zephyr"
            / "embedding_hitman_ratio_instances.csv",
        ),
        (
            "RQ3 Zephyr Hitman-ratio heatmap values",
            ref
            / "zephyr"
            / "embedding_hitman_ratio_heatmap_values_all_pa.csv",
            out
            / "figures"
            / "rq3_zephyr"
            / "embedding_hitman_ratio_heatmap_values_all_pa.csv",
        ),

        # Pegasus
        (
            "RQ3 Pegasus submissions",
            ref
            / "pegasus"
            / "embedding_submissions_averaged_over_attempts.csv",
            out
            / "figures"
            / "rq3_pegasus"
            / "embedding_submissions_averaged_over_attempts.csv",
        ),
        (
            "RQ3 Pegasus algorithm instances",
            ref
            / "pegasus"
            / "embedding_algorithm_instances.csv",
            out
            / "figures"
            / "rq3_pegasus"
            / "embedding_algorithm_instances.csv",
        ),
        (
            "RQ3 Pegasus heatmap values",
            ref
            / "pegasus"
            / "embedding_heatmap_values_all_pa.csv",
            out
            / "figures"
            / "rq3_pegasus"
            / "embedding_heatmap_values_all_pa.csv",
        ),
        (
            "RQ3 Pegasus Hitman-ratio instances",
            ref
            / "pegasus"
            / "embedding_hitman_ratio_instances.csv",
            out
            / "figures"
            / "rq3_pegasus"
            / "embedding_hitman_ratio_instances.csv",
        ),
        (
            "RQ3 Pegasus Hitman-ratio heatmap values",
            ref
            / "pegasus"
            / "embedding_hitman_ratio_heatmap_values_all_pa.csv",
            out
            / "figures"
            / "rq3_pegasus"
            / "embedding_hitman_ratio_heatmap_values_all_pa.csv",
        ),
    ]

    print()
    print("Comparing reproduced aggregated data with paper reference data")
    print("=============================================================")

    results = []

    for name, reference_path, reproduced_path in checks:
        results.append(
            compare_csv(
                name=name,
                reference_path=reference_path,
                reproduced_path=reproduced_path,
                rtol=args.rtol,
                atol=args.atol,
            )
        )

    print()
    print("=============================================================")

    passed = sum(results)
    total = len(results)

    if all(results):
        print(f"All checks passed ({passed}/{total}).")
        print(
            "The regenerated aggregated CSV files match the "
            "aggregated data used in the paper."
        )
        return

    failed = total - passed

    print(f"{failed} check(s) failed ({passed}/{total} passed).")
    sys.exit(1)


if __name__ == "__main__":
    main()