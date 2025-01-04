"""
Triangle shape derived class containing constant data

Author: JWT
"""
import numpy as np
from src.library.geoms.geometry import Geometry


class Triangle(Geometry):
    """
    @:brief Triangle geometry class to evaluate transformation and Jacobian with methods defined
    """

    def __init__(self, nodes, x, y):
        """
        @:brief            Main constructor for linear tri geometry
        :param nodes:      Node coordinates from the mesh parsed by reference
        :param x:          Array of quadrature zeros xi1-coordinates
        :param y:          Array of quadrature zeros xi2-coordinates
        """
        super().__init__()
        self.A = nodes[0]
        self.B = nodes[1]
        self.C = nodes[2]
        self.x = x
        self.y = y

    def ref_mapping(self):
        """
        @:brief Linear mapping between a reference tri element and a straight-sided tri element in real space
        :return: Arrays of x and y-coordinates in physical space given arrays in reference space
        """
        coords = np.empty((2, self.x.shape[0]), dtype=np.float64)
        for i in range(2):
            # coords[i] = 0.5 * (self.A[i] * (- self.x - self.y) +
            #             self.B[i] * (1. + self.x) +
            #             self.C[i] * (1. + self.y))
            coords[i] = (self.A[i] * (1. - self.x - self.y) +
                        self.B[i] * self.x +
                        self.C[i] * self.y)

        return coords

    def inv_mapping(self, coords):
        """
        @:brief Inverse mapping between a tri element in real space and a reference tri element
        :return: Arrays of x and y-coordinates in reference space given arrays in physical space
        """

        return self.inv_jacobian() @ (coords - self.A)

    @property
    def dxdxi1(self):

        dxdxi1 = np.empty((2, self.x.shape[0]), dtype=np.float64)
        for i in range(2):
            # dxdxi1[i] = 0.5 * (-self.A[i] + self.B[i])
            dxdxi1[i] = -self.A[i] + self.B[i]

        return dxdxi1

    @property
    def dxdxi2(self):

        dxdxi2 = np.empty((2, self.x.shape[0]), dtype=np.float64)
        for i in range(2):
            # dxdxi2[i] = 0.5 * (-self.A[i] + self.C[i])
            dxdxi2[i] = -self.A[i] + self.C[i]

        return dxdxi2

    @property
    def jacobian(self):
        """
        @:brief Jacobian calculation routines
        :return: 2D Jacobian based on linear mapping above
        """
        return np.array([[self.dxdxi1.T[i], self.dxdxi2.T[i]] for i in range(len(self.dxdxi1[0]))])

    def inv_jacobian(self):
        return np.linalg.inv(self.jacobian)

    def det_jacobian(self):
        return np.linalg.det(self.jacobian)
