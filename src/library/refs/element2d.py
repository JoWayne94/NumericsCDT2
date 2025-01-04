"""
Abstract base class for a two-dimensional reference element

Author: JWT
"""
from abc import ABC, abstractmethod
from src.library.refs.element import RefElem as Elem


class RefElem2d(Elem, ABC):
    """
    @:brief Abstract subclass to represent a 2d element in reference space
    """

    @property
    @abstractmethod
    def p2(self):
        """
        @:brief p2: Polynomial order in the xi_2 direction
        """
        pass

    @abstractmethod
    def get_quadrature_zeros_weights(self):
        """
        @:brief Quadrature zeros and weights to be stored as const data, overloaded depending on shapes
        :return: Quadrature point coordinates in [-1, 1] and weights
        """
        return None, None
