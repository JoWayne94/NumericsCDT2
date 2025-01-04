"""
Mesh data needed to do the continuous Galerkin discretisation

Author: JWT
"""
import numpy as np
from src.library.elems.seg import Segment as Seg
from src.library.elems.quad import Quadrilateral as Quad
from src.library.elems.tri import Triangle as Tri


class Mesh:
    """
    @:brief Contains information of the mesh
    """
    class MeshConnectivityData:
        """
        @:brief Mesh connectivity data subclass
        """
        def __init__(self):
            self.nodes = None
            self.ien = None
            self.ids = None
            self.boundaries = None
            self.n_dims = None

    class MeshDerivedData:
        """
        @:brief Mesh derived data subclass
        """
        def __init__(self):
            self.node_neighbours = None
            self.faces = None  # Internal faces

    def __init__(self, nodes, ien, ids, boundaries, n_dims, shape):
        """
        @:brief             Main constructor
        :param nodes:       Coordinates of all nodes stored in a numpy array
        :param ien:         IEN stored in a numpy array
        :param ids:         IDs stored in a numpy array
        :param boundaries:  Dictionary of boundary patches for the simulation
        :param n_dims:      Number of spatial dimensions
        :param shape:       Shape of the elements
        """
        self.connectivity_data = self.MeshConnectivityData()
        self.derived_data = self.MeshDerivedData()
        self.elements = None  # element objects stored in a list
        self.shape = shape

        # Connectivity data
        self.connectivity_data.nodes = nodes
        self.connectivity_data.ien = ien
        self.connectivity_data.ids = ids
        self.connectivity_data.boundaries = boundaries
        self.connectivity_data.n_dims = n_dims

    @classmethod
    def construct_from_mesh_input(cls, nodes_file, ien_file, bdry_file):
        """
        @:brief Read data from mesh directory and create mesh for simulation

        :param nodes_file:   Location of nodes file in machine
        :param ien_file:     Location of IEN file in machine
        :param bdry_file:    Location of bdry file in machine

        :return: Alternative main constructor with data from gmsh
        """
        print("Creating mesh... \n")

        nodes = np.loadtxt(nodes_file)
        ien = np.loadtxt(ien_file, dtype=np.int64)
        boundary_nodes = np.loadtxt(bdry_file, dtype=np.int64)

        south_boundary = np.where(nodes[boundary_nodes, 1] <= 110000.)[0]

        # Make all boundary points Dirichlet
        ID = np.zeros(len(nodes), dtype=np.int64)
        boundaries = dict()  # hold the boundary values
        n_eq = 0
        for i in range(len(nodes)):
            if i in south_boundary:
                ID[i] = -1
                boundaries[i] = 0.  # Dirichlet BC
            else:
                ID[i] = n_eq
                n_eq += 1
        mat_dim = np.max(ID) + 1
        n_e = ien.shape[0]
        N_nodes = nodes.shape[0]
        n_dims = nodes.shape[1]
        # Location matrix
        lm = np.zeros_like(ien.T)
        for e in range(n_e):
            for a in range(ien.shape[1]):
                lm[a, e] = ID[ien[e, a]]

        # pointlabels, points = cls.readPointsFromPolyMesh(polyMeshLocation, ndims)
        # boundary = cls.readBoundaryFromPolyMesh(polyMeshLocation)

        return cls(nodes, ien, ID, boundaries, n_dims, 'T')

    def construct_elements(self, n_elems, n_vars, source, p1, p2=0):
        """
        @:brief Create node-based elements only after nodes and ID arrays are created
        :param n_elems:          Number of elements
        :param n_vars:           Number of state variables
        :param source:           Source term to be incorporated in the force vector
        :param p1:               Polynomial order in the x-direction
        :param p2:               Polynomial order in the y-direction (default=0, 1D)
        :return: Return element objects array for the mesh object
        """
        self.elements, self.derived_data.node_neighbours, \
            self.derived_data.faces = self.elems_data_construction(n_elems, n_vars, source, p1, p2)

        print("Mesh created. \n")

    def elems_data_construction(self, n_elems, n_vars, source, p1, p2):
        """
        @:brief         Process data from mesh generator and create corresponding containers
        :param n_elems: Number of elements
        :param n_vars:  Number of state variables
        :param source:  Source term to be incorporated in the force vector
        :param p1:      Polynomial order in the x-direction
        :param p2:      Polynomial order in the y-direction (default=0, 1D)
        :return:        Arrays of element objects, node neighbours (element labels surrounding a node), and faces
        """
        vertices_list = [[n for n in v] for v in self.connectivity_data.ien]

        # Initialise end result: List of element labels corresponding to the neighbours of each node
        node_neighbours_list = []
        # Iterate through all nodes
        for i in range(self.connectivity_data.nodes.shape[0]):
            # Go through all elements in the vertices list. If list contains current node label, append element label
            tmp = [x for x, v in enumerate(vertices_list, 0) if v.count(i) > 0]
            node_neighbours_list.append(tmp)

        # Initialise end result: faces list
        faces_list = []
        if self.connectivity_data.n_dims == 1:
            # Iterate through all nodes with neighbouring element labels
            for index in range(len(node_neighbours_list)):
                # Ignore node if it only has one neighbour; they don't contribute to internal faces
                if len(node_neighbours_list[index]) == 1:
                    continue
                else:
                    faces_list.append([index, set(node_neighbours_list[index])])
        else:
            # Iterate through all nodes with neighbouring element labels
            for index in range(len(node_neighbours_list)):
                # Initialise temporary faces list
                tmp_face_list = []
                # Ignore node if it only has one neighbour; they don't contribute to internal faces
                if len(node_neighbours_list[index]) == 1:
                    continue
                else:
                    # List comprehension to go through all remaining nodes below current node. If contains common
                    # element labels (> 1), add common element labels to list
                    common_neighbour = [self.common_member(node_neighbours_list[index], point) for point in
                                       node_neighbours_list[index + 1:]]
                    # Iterate through common element labels list
                    # https://www.geeksforgeeks.org/python-non-none-elements-indices/
                    for enum, val in enumerate(common_neighbour):
                        # If contains common neighbouring element labels, append [node label who shares common
                        # neighbouring labels with current node, common neighbouring element labels] to temporary list
                        if val is not None:
                            tmp_face_list.append([enum + index + 1, val])
                    # If temporary list contains something, iterate through the list
                    if len(tmp_face_list) != 0:
                        for k in range(len(tmp_face_list)):
                            # Append [[current node label, node label who shares common neighbouring element labels],
                            # common element labels in a set] to final faces list
                            faces_list.append([[index, tmp_face_list[k][0]], tmp_face_list[k][1]])

        if self.connectivity_data.n_dims == 1:
            # Initialise all element objects array
            elements = np.empty(n_elems, dtype=Seg)
            # Iterate over all elements
            for i in range(n_elems):
                # Iterate through all faces, if current element label corresponds to an internal face, add face label to list
                elem_in_face_list = [x for x, j in enumerate(faces_list, 0) if len({i}.intersection(j[1])) > 0]
                # Element neighbours labels list
                neighbours_list = []
                # Iterate over all vertices of a shape
                for vertex in range(2):
                    # Iterate over all internal faces corresponding to current element, if vertex labels correspond to
                    # node labels that make up said internal face, store neighbouring element label to list
                    neighbours_list.append(next(iter([list(faces_list[face][1] ^ {i})[0] for face in elem_in_face_list if
                                                     len({vertices_list[i][vertex]}.intersection(
                                                         {faces_list[face][0]})) > 0]), None))

                elements[i] = Seg(self.connectivity_data.ien[i],
                                  np.array([[self.connectivity_data.nodes[i]], [self.connectivity_data.nodes[i + 1]]]),
                                  neighbours_list, n_vars, source, p1)
        elif self.connectivity_data.n_dims == 2:
            if self.shape == "Q":
                # Initialise all element objects array
                elements = np.empty(n_elems, dtype=Quad)
                # Iterate over all cells
                for i in range(n_elems):
                    # Iterate through all faces, if current element label corresponds to an internal face, add face label to list
                    elem_in_face_list = [x for x, j in enumerate(faces_list, 0) if len({i}.intersection(j[1])) > 0]
                    # Element neighbours labels list
                    neighbours_list = []
                    # Iterate over all vertices of a shape
                    for vertex in range(4):  # 3 if 2D tri, 4 if 2D quad
                        next_vertex = vertex + 1
                        if next_vertex == 4:  # 3 if 2D tri, 4 if 2D quad
                            next_vertex = 0  # go back to first vertex

                        # Iterate over all internal faces corresponding to current element, if vertex labels correspond
                        # to node labels that make up said internal face, store neighbouring element label to list
                        neighbours_list.append(next(iter([list(faces_list[face][1] ^ {i})[0] for face in elem_in_face_list if
                                                         len({vertices_list[i][vertex],
                                                              vertices_list[i][next_vertex]}.intersection(
                                                             set(faces_list[face][0]))) > 1]), None))

                    elements[i] = Quad(self.connectivity_data.ien[i],
                                  self.connectivity_data.nodes[self.connectivity_data.ien[i]],
                                  neighbours_list, n_vars, source, p1, p2)
            elif self.shape == "T":
                # Initialise all element objects array
                elements = np.empty(n_elems, dtype=Tri)
                # Iterate over all cells
                for i in range(n_elems):
                    # Iterate through all faces, if current element label corresponds to an internal face, add face label to list
                    elem_in_face_list = [x for x, j in enumerate(faces_list, 0) if len({i}.intersection(j[1])) > 0]
                    # Element neighbours labels list
                    neighbours_list = []
                    # Iterate over all vertices of a shape
                    for vertex in range(3):  # 3 if 2D tri, 4 if 2D quad
                        next_vertex = vertex + 1
                        if next_vertex == 3:  # 3 if 2D tri, 4 if 2D quad
                            next_vertex = 0  # go back to first vertex

                        # Iterate over all internal faces corresponding to current element, if vertex labels correspond
                        # to node labels that make up said internal face, store neighbouring element label to list
                        neighbours_list.append(next(iter([list(faces_list[face][1] ^ {i})[0] for face in elem_in_face_list if
                                                         len({vertices_list[i][vertex],
                                                              vertices_list[i][next_vertex]}.intersection(
                                                             set(faces_list[face][0]))) > 1]), None))

                    elements[i] = Tri(self.connectivity_data.ien[i],
                                  self.connectivity_data.nodes[self.connectivity_data.ien[i]],
                                  neighbours_list, n_vars, source, p1, p2)
            else:
                raise NotImplementedError("Element shape not implemented.")
        else:
            raise NotImplementedError("No. of spatial dimensions not implemented.")

        return elements, node_neighbours_list, faces_list

    def common_member(self, a, b):
        """
        https://www.geeksforgeeks.org/python-check-two-lists-least-one-element-common/
        :param a: First list input
        :param b: Second list input
        :return:  Common elements if number of common elements > 1, else None
        """
        a_set = set(a)
        b_set = set(b)
        intersect = a_set.intersection(b_set)
        if len(intersect) > self.connectivity_data.n_dims - 1:
            return intersect
        pass
