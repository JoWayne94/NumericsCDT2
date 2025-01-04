"""
Individual element object class/struct containing element and geometry data.

Abstract base class for extensions of different calculation routines in child classes.

Author: JWT
"""
import numpy as np
from abc import ABC, abstractmethod


class Element(ABC):
    """
    @:brief Contains information of a single element
    """
    class ElemGeometryData:
        """
        @:brief Node-based shape data subclass
        """
        def __init__(self):
            self.shape = None
            self.ien = None
            self.nodes = None
            self.neighbour_labels = None

    class ElemCalc:
        """
        @:brief Calculation routines subclass
        """
        def __init__(self):
            self.vol = None
            self.elem_centre = None

    def __init__(self, ien, nodes, neighbour_labels):
        """
        @:brief               Main constructor
        :param ien:           Node IEN making up this element stored in a list
        :param nodes:         Node coordinates from mesh supposedly parsed in here by reference/assignment
        :param neighbour_labels: Neighbouring element labels, starting from first face of element.
                                 If no neighbour corresponding to that face, a None value is stored
        """
        self.geom_data = self.ElemGeometryData()
        self.calculations = self.ElemCalc()

        # Geometry data
        self.geom_data.ien = ien
        self.geom_data.nodes = nodes
        self.geom_data.neighbour_labels = neighbour_labels

        # Calculation routines
        """
        :param vol:          Element volume
        :param elem_centre:  Element centre
        :param face_normals: Face unit normal vectors
        """
        self.calculations.vol = self.calculate_elem_vol()
        self.calculations.elem_centre = self.calculate_elem_centre()
        self.calculations.face_normals = self.calculate_face_normals()

        # print("Element created.\n")

    def calculate_elem_centre(self):
        """
        @:brief  Average of coordinates of nodes
        :return: Element centre coordinates of this element
        """
        # Initialise number of nodes that make up the element
        n_nodes = 0
        # Initialise end result: (x, y) coordinates
        result = np.zeros(len(self.geom_data.nodes[0]))

        # For every point in cell
        for i in range(len(self.geom_data.ien)):
            # Add up the coordinates of points
            result += self.geom_data.nodes[i]
            # Counter for numbers of points in cell
            n_nodes += 1

        # Divide sums of coordinates by number of points in cell
        result /= n_nodes

        return result

    @abstractmethod
    def calculate_elem_vol(self):
        """
        @:brief  Calculates volume of an element
        :return: Volume of the element
        """
        pass

    @abstractmethod
    def calculate_face_normals(self):
        """
        @:brief  Calculate unit normal vector of faces
        :return: Unit normal vector ([number of faces, number of face Gauss points, vector])
        """
        pass
