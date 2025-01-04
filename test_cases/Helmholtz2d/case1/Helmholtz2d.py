"""
Test cases for the 2d Helmholtz equation

Author: JWT
"""

# Import third-party libraries
import sys, os
import matplotlib.pyplot as plt

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

# Import mesh class
from src.library.mesh.mesh import Mesh

# Import equation types
from src.solvers.Helmholtz import *


def source1(coords):
    return 1.

def exact1(coords):
    return coords[:, 0] * (1. - coords[:, 0] / 2.)

def source2(coords):
    return (2. * coords[0] * (coords[0] - 2.) * (3. * coords[1]**2 - 3. * coords[1] + 0.5)
            + coords[1]**2 * (coords[1] - 1.)**2)

def exact2(coords):
    return coords[:, 0] * (1. - coords[:, 0] / 2.) * coords[:, 1]**2 * (1. - coords[:, 1])**2

def source3(coords):
    return 1. - coords[0]

def exact3(coords):
    return coords[:, 0] / 6. * (coords[:, 0]**2 - 3. * coords[:, 0] + 3.)

def source4(coords):
    return (1. - coords[0])**2

def exact4(coords):
    return coords[:, 0] / 12. * (4. - 6. * coords[:, 0] + 4. * coords[:, 0]**2 - coords[:, 0]**3)


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
    x_left = DOMAIN_BOUNDARIES[0][0]
    x_right = DOMAIN_BOUNDARIES[1][0]
    y_bottom = DOMAIN_BOUNDARIES[0][1]
    y_top = DOMAIN_BOUNDARIES[2][1]
    line_styles = ['r.', 'gv', 'b<', 'k*', 'ro', 'g^', 'b>', 'ks']
    plt.rc('text', usetex=True)

    """ Problem definition """
    source = source4
    exact = exact4
    nodes, IEN, ID, boundaries, lm, mat_dim = mesh_gen(NO_OF_NODES, DOMAIN_BOUNDARIES,
                                                       BOUNDARY_CONDITIONS_TYPE, BOUNDARY_CONDITIONS_VALUES, ELEM_SHAPE)

    # print(f"nodes:\n{nodes}")
    # print(f"IEN:\n{IEN}")
    # print(f"ID:\n{ID}")
    # print(f"boundaries:\n{boundaries}")
    # print(f"lm:\n{lm}")

    # plot_2d_quad_mesh(nodes, IEN, lw)
    # plot_2d_tri_mesh(nodes, IEN, lw)

    cg = Mesh(nodes, IEN, ID, boundaries, NO_OF_DIMENSIONS, ELEM_SHAPE)
    cg.construct_elements(NO_OF_ELEMS, NO_OF_VARIABLES, source, P1, P2)

    lhs, rhs, u = solve(cg, lm, NO_OF_ELEMS, P1, P2)
    # print(f"Solutions:\n{u}")
    # print(f"Global Laplacian matrix:\n{lhs}")
    # print(f"Global force vector:\n{rhs}")

    """ Analytical solution """
    u_exact = exact(nodes)

    plot_2d_solutions(u, u_exact, NO_OF_NODES[0], NO_OF_NODES[1], x_left, x_right, y_bottom, y_top)
