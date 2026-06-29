from dataclasses import dataclass, field
from typing import Any, FrozenSet, List, Dict
from collections import defaultdict
from src.core.problem import HittingSetProblem


@dataclass
class SimulatorResult:
    """
    Represents a single sample returned by a simulator.

    Each result stores the sampled solution as an immutable set of selected
    elements, its associated energy, and optionally the number of times the
    same sample was observed during the simulation.
    """
    sample: FrozenSet[int]
    energy: float
    num_occurrences: int | None = None


@dataclass
class SimulatorResultSet:
    """
    Collects the full set of results produced by a simulator.

    It contains the processed simulator samples, an optional raw backend result,
    and a metadata dictionary that can store additional execution details such
    as solver parameters, timing information, or run configuration.
    """
    results: List[SimulatorResult]
    raw_result: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LengthStatistics:
    """
    Stores statistical information for solutions of a fixed length.

    For a given solution size, this class records how many true minimal hitting
    sets exist, how many were found by the algorithm, the corresponding recovery
    ratio, and several frequency-based quantities derived from the sampled results.
    """
    length: int
    true_count: int
    found_count: int
    found_ratio: float
    total_frequency: float
    mean_frequency_found: float
    mean_frequency_all_true: float


@dataclass(slots=True)
class StatisticalResult:
    """ convenience
    Stores the global and per-length statistics computed for an algorithm run.

    It includes the total number of true minimal hitting sets, the number of
    recovered solutions, the global recovery ratio, and a mapping from each
    solution length to the corresponding length-specific statistics.
    """
    total_true_mhs: int
    total_found_mhs: int
    global_found_ratio: float
    by_length: Dict[int, LengthStatistics] = field(default_factory=dict)

    def lengths(self) -> List[int]:
        """
        Return the list of solution lengths for which statistics are available.
        """
        return list(self.by_length.keys())


@dataclass
class AlgorithmResult:
    """
    Represents a single solution returned by an algorithm.

    The solution is stored as an immutable set of selected elements, together
    with an optional occurrence count when the algorithm is based on repeated
    sampling or multiple reads.
    """
    set: FrozenSet[int]
    num_occurrences: int | None = None


@dataclass
class AlgorithmResultSet:
    """
    Represents the full set of solutions obtained for a given problem instance.

    This class stores the reference hitting set problem, the list of solutions
    found by the algorithm, and optional metadata describing the execution.
    It also provides utility methods for computing aggregate recovery stati conveniencestics
    with respect to the known minimal hitting sets of the problem.
    """
    problem: HittingSetProblem
    num_qubits: int
    num_couplers: int
    solutions: List[AlgorithmResult]
    metadata: Dict[str, Any] = field(default_factory=dict)
    mode: str = 'minimal'

    def get_total_count(self, length: int) -> int:
        """
        Return the total number of samples or reads used to normalize the
        occurrence counts of solutions of the given length.

        The default implementation uses the number of reads stored in metadata.
        Subclasses may override this method when the normalization depends on
        the solution length or on a different counting rule.
        """
        return self.metadata["num_reads"]

    def compute_statistics(self, output: bool = False) -> StatisticalResult:
        """
        Compute global and per-length statistics for the solutions found by
        the algorithm.

        The method compares the recovered solutions against the known minimal
        hitting sets of the problem, groups them by solution length, and
        evaluates several recovery and frequency-based metrics.

        If output is True, the statistics are also printed in a readable form.

        Returns:
            A StatisticalResult object containing the global summary and the
            detailed statistics for each solution length.
        """
        if output:
            print("Computing statistics...")

        true_mhs = [frozenset(mhs) for mhs in self.problem.minimal_hitting_sets]

        # If minimum mode select only minimum hitting sets
        if self.mode == 'minimum':

            if not true_mhs:
                raise ValueError(
                    "No reference minimal hitting sets are available. "
                    "Cannot compute found ratio or minimum-cardinality statistics."
                )

            minimum_len = min([len(x) for x in true_mhs])
            true_mhs = [x for x in true_mhs if len(x) == minimum_len]
        total_true_mhs = len(true_mhs)

        true_by_length: dict[int, list[frozenset[int]]] = defaultdict(list)
        for mhs in true_mhs:
            true_by_length[len(mhs)].append(mhs)

        found_by_length: dict[int, list] = defaultdict(list)
        for sol in self.solutions:
            found_by_length[len(sol.set)].append(sol)

        total_found_mhs = len(self.solutions)

        by_length: dict[int, LengthStatistics] = {}

        for length in sorted(true_by_length.keys()):
            true_count = len(true_by_length[length])

            total_count = self.get_total_count(length)
            if total_count <= 0:
                raise ValueError(f"total_count must be positive for length {length}, got {total_count}")

            solutions_of_length = found_by_length.get(length, [])

            found_count = len(solutions_of_length)
            found_ratio = found_count / true_count if true_count > 0 else 0.0

            frequencies = [sol.num_occurrences / total_count for sol in solutions_of_length]
            total_frequency = sum(frequencies)

            mean_frequency_found = (
                total_frequency / found_count if found_count > 0 else 0.0
            )

            mean_frequency_all_true = (
                total_frequency / true_count if true_count > 0 else 0.0
            )

            by_length[length] = LengthStatistics(
                length=length,
                true_count=true_count,
                found_count=found_count,
                found_ratio=found_ratio,
                total_frequency=total_frequency,
                mean_frequency_found=mean_frequency_found,
                mean_frequency_all_true=mean_frequency_all_true,
            )

            if output:
                print(f"\nLength {length}")
                print(f"  true_count               = {true_count}")
                print(f"  found_count              = {found_count}")
                print(f"  found_ratio              = {found_ratio:.4f}")
                print(f"  total_frequency          = {total_frequency:.4f}")
                print(f"  mean_frequency_found     = {mean_frequency_found:.4f}")
                print(f"  mean_frequency_all_true  = {mean_frequency_all_true:.4f}")

        global_found_ratio = (
            total_found_mhs / total_true_mhs if total_true_mhs > 0 else 0.0
        )
        if self.mode == 'minimum':
            global_found_ratio = by_length[minimum_len].found_ratio

        if output:
            print("\nGlobal statistics")
            print(f"  total_true_mhs     = {total_true_mhs}")
            print(f"  total_found_mhs    = {total_found_mhs}")
            print(f"  global_found_ratio = {global_found_ratio:.4f}")

        return StatisticalResult(
            total_true_mhs=total_true_mhs,
            total_found_mhs=total_found_mhs,
            global_found_ratio=global_found_ratio,
            by_length=by_length,
        )