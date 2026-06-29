from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


GLOBAL_PATTERN = re.compile(r"^global_stats_(.+)\.csv$")
LENGTH_PATTERN = re.compile(r"^length_stats_(.+)\.csv$")


def extract_seed_from_filename(path: Path, pattern: re.Pattern[str]) -> str:
    """
    Extract the seed part from filenames such as:

        global_stats_42.csv
        length_stats_42.csv

    The returned seed is kept as a string because some seeds may not be purely
    numeric.
    """
    match = pattern.match(path.name)

    if match is None:
        raise ValueError(f"Could not extract seed from filename: {path.name}")

    return match.group(1)


def load_and_tag_csv(path: Path, seed: str) -> pd.DataFrame:
    """
    Load one CSV file and add a seed column.

    If the file already contains a seed column, it is overwritten using the seed
    parsed from the filename, because the filename is the safest source here.
    """
    df = pd.read_csv(path)
    df["seed"] = seed
    return df


def concatenate_seed_files(
    input_dir: str | Path,
    output_dir: str | Path | None = None,
    overwrite: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge all global_stats_{seed}.csv and length_stats_{seed}.csv files into:

        global_stats.csv
        length_stats.csv

    Parameters
    ----------
    input_dir:
        Directory containing the per-seed CSV files.

    output_dir:
        Directory where the merged CSV files should be written.
        If None, uses input_dir.

    overwrite:
        If False, raises an error if global_stats.csv or length_stats.csv
        already exist.

    Returns
    -------
    global_df, length_df:
        The merged DataFrames.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir is not None else input_dir

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    global_output_path = output_dir / "global_stats.csv"
    length_output_path = output_dir / "length_stats.csv"

    if not overwrite:
        if global_output_path.exists():
            raise FileExistsError(f"Output file already exists: {global_output_path}")

        if length_output_path.exists():
            raise FileExistsError(f"Output file already exists: {length_output_path}")

    global_files = sorted(
        path
        for path in input_dir.glob("global_stats_*.csv")
        if path.name != "global_stats.csv"
    )

    length_files = sorted(
        path
        for path in input_dir.glob("length_stats_*.csv")
        if path.name != "length_stats.csv"
    )

    if not global_files:
        raise FileNotFoundError(
            f"No files matching global_stats_{{seed}}.csv found in {input_dir}"
        )

    if not length_files:
        raise FileNotFoundError(
            f"No files matching length_stats_{{seed}}.csv found in {input_dir}"
        )

    global_frames: list[pd.DataFrame] = []
    length_frames: list[pd.DataFrame] = []

    global_seeds: set[str] = set()
    length_seeds: set[str] = set()

    for path in global_files:
        seed = extract_seed_from_filename(path, GLOBAL_PATTERN)
        global_seeds.add(seed)
        global_frames.append(load_and_tag_csv(path, seed))

    for path in length_files:
        seed = extract_seed_from_filename(path, LENGTH_PATTERN)
        length_seeds.add(seed)
        length_frames.append(load_and_tag_csv(path, seed))

    missing_length = sorted(global_seeds - length_seeds)
    missing_global = sorted(length_seeds - global_seeds)

    if missing_length:
        print(
            "Warning: found global_stats files without matching length_stats files "
            f"for seeds: {missing_length}"
        )

    if missing_global:
        print(
            "Warning: found length_stats files without matching global_stats files "
            f"for seeds: {missing_global}"
        )

    global_df = pd.concat(global_frames, ignore_index=True)
    length_df = pd.concat(length_frames, ignore_index=True)

    global_df.to_csv(global_output_path, index=False)
    length_df.to_csv(length_output_path, index=False)

    print(f"Written: {global_output_path}")
    print(f"Rows: {len(global_df)}")
    print(f"Seeds: {len(global_seeds)}")

    print(f"Written: {length_output_path}")
    print(f"Rows: {len(length_df)}")
    print(f"Seeds: {len(length_seeds)}")

    return global_df, length_df


if __name__ == "__main__":
    # Change this to your benchmark directory.
    BASEPATH = Path(
        "/home/v3r7ux/Documents/University/Porto/Internship/QuantumAnnealer/output/benchmark/classical_annealing/minimal/scalability"
    )

    concatenate_seed_files(
        input_dir=BASEPATH,
        output_dir=BASEPATH,
        overwrite=True,
    )