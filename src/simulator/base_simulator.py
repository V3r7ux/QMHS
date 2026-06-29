from abc import ABC, abstractmethod
from typing import Dict, Tuple


class BaseSimulator(ABC):
    """
    Abstract base class for all simulator backends.

    A simulator is responsible for running a QUBO model and returning results
    in a common format. Subclasses must implement the logic for executing the
    simulation and for converting raw samples into solution sets.
    """

    @abstractmethod
    def get_set_from_sample(self, sample):
        """
        Convert a raw sample returned by the simulator into a solution set.
        """
        pass
    
    @abstractmethod
    def simulate(self, qubo: Dict[Tuple[int, int], float]):
        """
        Run the QUBO simulation and return the corresponding simulator results.
        """
        pass