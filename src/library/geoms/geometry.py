"""
Abstract base class for various shapes. Contains constant data

Author: JWT
"""
from abc import ABC, abstractmethod


class Geometry(ABC):

    def __init__(self):
        """
        @:brief Main constructor
        """
        pass

    @abstractmethod
    def ref_mapping(self):
        pass

    @abstractmethod
    def det_jacobian(self):
        pass
