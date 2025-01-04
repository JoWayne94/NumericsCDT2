"""
2d Poisson equation on the UK grid

Author: JWT
"""

# Import third-party libraries
import sys, os

import matplotlib.pyplot as plt
import numpy as np

# Import user settings
from setup import *

# Configure system path
path = os.path.dirname(__file__)

if (sys.platform[:3] == 'win') or (sys.platform[:3] == 'Win'):
    sys.path.append(os.path.abspath(os.path.join(path, '..\..\..')))
else:
    sys.path.append(os.path.abspath(os.path.join(path, '../../..')))

# Import mesh generator
from mesh.meshGen import *

# Import equation types
from src.solvers.Helmholtz import *


def source1(coords):
    return 1.

def source2(coords, std=1000.):
    return np.exp(-1. / (2. * std ** 2) * ((coords[0] - soton[0]) ** 2 + (coords[1] - soton[1]) ** 2))


if __name__ == '__main__':
    """
    main()
    """

    # if len(sys.argv) < 2:
    if EQUATION_TYPE == 'Helmholtz':
        solve = helmholtz_operator
    else:
        raise NotImplementedError('Equation type is not implemented.')

    if NO_OF_DIMENSIONS == 1:
        mesh_gen = generate_1d_mesh
        ELEM_SHAPE = "S"
        NO_OF_ELEMS = NO_OF_NODES[0] - 1
    elif NO_OF_DIMENSIONS == 2:
        mesh_gen = generate_2d_mesh
        NO_OF_ELEMS = (NO_OF_NODES[0] - 1) * (NO_OF_NODES[1] - 1)
        if ELEM_SHAPE == 'T': NO_OF_ELEMS *= 2
    else:
        raise NotImplementedError('Number of spatial dimensions not supported.')

    """ Plotting parameters and visualisations """
    lw = 1.5
    ms = 5.
    line_styles = ['r.', 'gv', 'b<', 'k*', 'ro', 'g^', 'b>', 'ks']
    plt.rc('text', usetex=True)

    """ Problem definition """
    source = source2
    soton = np.array([442365., 115483])
    reading = np.array([473993., 171625.])

    if USER_INPUT:
        nodes = np.loadtxt(sys.path[-1] + f'/mesh/las_grids/las_nodes_{INPUT_NAME}k.txt')
        IEN = np.loadtxt(sys.path[-1] + f'/mesh/las_grids/las_IEN_{INPUT_NAME}k.txt', dtype=np.int64)
        boundary_nodes = np.loadtxt(sys.path[-1] + f'/mesh/las_grids/las_bdry_{INPUT_NAME}k.txt',
                                    dtype=np.int64)

        # Make all boundary points Dirichlet
        ID = np.zeros(len(nodes), dtype=np.int64)
        boundaries = dict()  # hold the boundary values
        n_eq = 0
        for i in range(len(nodes[:, 1])):
            if i in boundary_nodes:
                ID[i] = -1
                boundaries[i] = 0.  # Dirichlet BC
            else:
                ID[i] = n_eq
                n_eq += 1
        mat_dim = np.max(ID) + 1
        NO_OF_ELEMS = IEN.shape[0]
        N_nodes = nodes.shape[0]
        NO_OF_DIMENSIONS = nodes.shape[1]
        # Location matrix
        lm = np.zeros_like(IEN.T)
        for e in range(NO_OF_ELEMS):
            for a in range(IEN[0].shape[0]):
                lm[a, e] = ID[IEN[e, a]]

        # plot_uk_mesh(nodes, boundary_nodes, IEN, lw)

    else:
        nodes, IEN, ID, boundaries, lm, mat_dim = mesh_gen(NO_OF_NODES, DOMAIN_BOUNDARIES,
                                                           BOUNDARY_CONDITIONS_TYPE,
                                                           BOUNDARY_CONDITIONS_VALUES, ELEM_SHAPE)

    # print(f"nodes:\n{nodes}")
    # print(f"IEN:\n{IEN}")
    # print(f"ID:\n{ID}")
    # print(f"boundaries:\n{boundaries}")
    # print(f"lm:\n{lm}")
    # plot_2d_tri_mesh(nodes, IEN, lw)

    cg = Mesh(nodes, IEN, ID, boundaries, NO_OF_DIMENSIONS, ELEM_SHAPE)
    cg.construct_elements(NO_OF_ELEMS, NO_OF_VARIABLES, source, P1, P2)

    lhs, rhs, u = solve(cg, lm, NO_OF_ELEMS, P1, P2, KAPPA)
    # print(f"Solutions:\n{u}")
    # print(f"Global Laplacian matrix:\n{lhs}")
    # print(f"Global force vector:\n{rhs}")

    """ Visualisation """
    x_left = np.min(nodes[:, 0])
    x_right = np.max(nodes[:, 0])
    y_bottom = np.min(nodes[:, 1])
    y_top = np.max(nodes[:, 1])

    u /= np.max(u)

    reading_elem = np.array([0], dtype=np.int32)
    reading_local_coords = np.zeros((1, 2), dtype=np.float64)
    for elem in range(NO_OF_ELEMS):
        reading_local_coords = cg.elements[elem].geoms.geometry.inv_mapping(reading)
        if reading_local_coords[0][0] >= 0. and reading_local_coords[0][1] >= 0. and np.sum(reading_local_coords[0]) < np.sqrt(2.):
            reading_elem[0] = elem
            break

    reading_u = (u[IEN[reading_elem[0], 0]] * (1. - reading_local_coords[0][0] - reading_local_coords[0][1]) +
                 u[IEN[reading_elem[0], 1]] * reading_local_coords[0][0] +
                 u[IEN[reading_elem[0], 2]] * reading_local_coords[0][1])

    print(reading_u)

    plt.close('all')
    plot_uk_solutions(nodes[:, 0], nodes[:, 1], u, IEN, x_left, x_right, y_bottom, y_top,
                    f"$N_e = {NO_OF_ELEMS}$, pure diffusion case", ms, soton, reading, r_e=reading_elem)
