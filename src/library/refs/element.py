"""
Abstract base class for a reference element. Contains constant data

Author: JWT
"""
from abc import ABC, abstractmethod


class RefElem(ABC):
    """
    @:brief Base class to represent an element in reference space
    """

    @property
    @abstractmethod
    def p1(self):
        """
        @:brief p1: Polynomial order in the xi_1 direction
        """
        pass

    @p1.setter
    @abstractmethod
    def p1(self, value):
        pass

    @property
    @abstractmethod
    def quadrature(self):
        """
        @:brief quadrature: Type of quadrature selected
        """
        pass

    @property
    @abstractmethod
    def zeros(self):
        """
        @:brief zeros: Zeros of polynomials
        """
        pass

    @property
    @abstractmethod
    def weights(self):
        """
        @:brief weights: Quadrature weights
        """
        pass
