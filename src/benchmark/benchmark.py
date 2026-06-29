import csv
from dataclasses import dataclass, field, asdict
from pathlib import Path
from src.core.result import StatisticalResult


@dataclass(slots=True)
class BenchmarkGlobalRecord:
    """
    Stores the global benchmark statistics for a single problem and algorithm.

    Each record summarizes the overall recovery performance of one algorithm
    on one problem instance, including the problem size, the logical size of
    the largest QUBO generated during the run, and the total number of true
    and recovered minimal hitting sets.
    """
    problem_id: str
    algorithm: str
    universe_size: int
    num_test_cases: int
    num_qubits: int
    num_couplers: int
    total_true_mhs: int
    total_found_mhs: int
    global_found_ratio: float
    runtime_seconds: float


@dataclass(slots=True)
class BenchmarkLengthRecord:
    """
    Stores benchmark statistics restricted to solutions of a specific length.

    Each record contains the recovery and frequency-based metrics associated
    with one solution length for a given problem instance and algorithm.
    """
    problem_id: str
    algorithm: str
    universe_size: int
    num_test_cases: int
    length: int
    true_count: int
    found_count: int
    found_ratio: float
    total_frequency: float
    mean_frequency_found: float
    mean_frequency_all_true: float


@dataclass(slots=True)
class BenchmarkRecordSet:
    """
    Collects benchmark records for multiple problems and algorithms.

    The class stores both global statistics and length-resolved statistics,
    and provides methods to append new statistical results and export the
    accumulated records to CSV files for later analysis and plotting.
    """
    global_records: list[BenchmarkGlobalRecord] = field(default_factory=list)
    length_records: list[BenchmarkLengthRecord] = field(default_factory=list)

    def add_statistical_result(
        self,
        problem_id: str,
        algorithm: str,
        universe_size: int,
        num_test_cases: int,
        num_qubits: int,
        num_couplers: int,
        stats: "StatisticalResult",
        runtime_seconds: float,
    ) -> None:
        """
        Add the global and per-length statistics of one algorithm run to the benchmark.

        This method converts a StatisticalResult object into one global benchmark
        record and a set of length-specific records, then appends them to the
        corresponding internal collections.

        Parameters
        ----------
        num_qubits:
            Maximum number of logical variables used by the algorithm for this
            problem instance. For algorithms that scan multiple cardinalities,
            this should be the maximum value obtained across all generated QUBOs.

        num_couplers:
            Maximum number of logical couplers, namely non-zero off-diagonal QUBO
            edges, used by the algorithm for this problem instance. For algorithms
            that scan multiple cardinalities, this should be the maximum value
            obtained across all generated QUBOs.
        """
        self.global_records.append(
            BenchmarkGlobalRecord(
                problem_id=problem_id,
                algorithm=algorithm,
                universe_size=universe_size,
                num_test_cases=num_test_cases,
                num_qubits=num_qubits,
                num_couplers=num_couplers,
                total_true_mhs=stats.total_true_mhs,
                total_found_mhs=stats.total_found_mhs,
                global_found_ratio=stats.global_found_ratio,
                runtime_seconds=runtime_seconds,
            )
        )

        for length in stats.lengths():
            len_stats = stats.by_length[length]

            self.length_records.append(
                BenchmarkLengthRecord(
                    problem_id=problem_id,
                    algorithm=algorithm,
                    universe_size=universe_size,
                    num_test_cases=num_test_cases,
                    length=len_stats.length,
                    true_count=len_stats.true_count,
                    found_count=len_stats.found_count,
                    found_ratio=len_stats.found_ratio,
                    total_frequency=len_stats.total_frequency,
                    mean_frequency_found=len_stats.mean_frequency_found,
                    mean_frequency_all_true=len_stats.mean_frequency_all_true,
                )
            )

    def num_global_records(self) -> int:
        """
        Return the number of stored global benchmark records.
        """
        return len(self.global_records)

    def num_length_records(self) -> int:
        """
        Return the number of stored length-specific benchmark records.
        """
        return len(self.length_records)

    @staticmethod
    def _write_csv(path: Path, rows: list[dict]) -> None:
        """
        Write a list of dictionary rows to a CSV file.

        The method creates the output directory if needed and writes the CSV
        header automatically from the keys of the first row. If the input list
        is empty, no file is written.
        """
        if not rows:
            return

        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    def export_global_csv(self, path: str | Path) -> None:
        """
        Export all global benchmark records to a CSV file.
        """
        path = Path(path)
        rows = [asdict(record) for record in self.global_records]
        self._write_csv(path, rows)

    def export_length_csv(self, path: str | Path) -> None:
        """
        Export all length-specific benchmark records to a CSV file.
        """
        path = Path(path)
        rows = [asdict(record) for record in self.length_records]
        self._write_csv(path, rows)

    def export_csv(self, output_dir: str | Path) -> None:
        """
        Export both global and length-specific benchmark records to CSV files.

        The method writes two files in the given output directory:
        - global_stats.csv for global statistics
        - length_stats.csv for length-resolved statistics
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.export_global_csv(output_dir / "global_stats.csv")
        self.export_length_csv(output_dir / "length_stats.csv")