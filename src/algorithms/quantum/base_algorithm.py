from abc import ABC, abstractmethod


class BaseAlgorithm(ABC):
    """
    Abstract base class for all quantum algorithm implementations.

    A quantum algorithm is responsible for constructing the QUBO or Hamiltonian
    formulation of a problem and for executing the full solution pipeline
    using a given solver or simulator backend.
    """

    @abstractmethod
    def construct_hamiltonian(self, universe, test_cases, *args, **kwargs):
        """
        Build the Hamiltonian or QUBO representation associated with the problem.

        This method takes the problem data, such as the universe and the test
        cases, and returns the corresponding optimization model to be solved.
        """
        pass
    
    @abstractmethod
    def run(self, problem, solver):
        """
        Execute the quantum algorithm on the given problem using the provided solver.

        This method typically constructs the Hamiltonian, runs the solver,
        and returns the processed algorithm results.
        """
        pass