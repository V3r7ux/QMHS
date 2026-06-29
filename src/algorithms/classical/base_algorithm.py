from abc import ABC, abstractmethod


class BaseAlgorithm(ABC):
    """
    Abstract base class for all classical algorithm implementations.

    A classical algorithm is responsible for executing the full solution pipeline
    on a given problem.
    """

    @abstractmethod
    def run(self, problem):
        """
        Execute the classical algorithm on the given problem.

        This method returns the processed algorithm results.
        """
        pass