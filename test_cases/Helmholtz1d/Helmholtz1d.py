"""
Test cases for the 1d Helmholtz equation

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
    sys.path.append(os.path.abspath(os.path.join(path, '..\..')))
else:
    sys.path.append(os.path.abspath(os.path.join(path, '../..')))

# Import mesh generator
from mesh.meshGen import *

# Import mesh class
from src.library.mesh.mesh import Mesh

# Import equation types
from src.solvers.Helmholtz import *


def source1(x_coords):
    return 2.

def exact1(x_coords):
    return 1. - np.power(x_coords, 2)

def source2(x_coords):
    return 1. - x_coords

def exact2(x_coords):
    return x_coords * (np.power(x_coords, 2) - 3. * x_coords + 3.) / 6.

def source3(x_coords):
    return np.power(1. - x_coords, 2)

def exact3(x_coords):
    return x_coords * (4. - 6. * x_coords + 4. * np.power(x_coords, 2) - np.power(x_coords, 3)) / 12.

def source4(x_coords):
    if abs(x_coords - 0.5) < 0.25: return 1.
    else: return 0.

def exact4(x_coords):
    psi_exact = np.empty_like(x_coords)
    for i in range(len(x_coords)):
        if x_coords[i] < 0.25: psi_exact[i] = 0.3 * x_coords[i] + 0.1
        elif 0.25 <= x_coords[i] < 0.75: psi_exact[i] = -0.5 * x_coords[i] * x_coords[i] + 0.55 * x_coords[i] + (11. / 160.)
        else: psi_exact[i] = -0.2 * x_coords[i] + 0.35

    return psi_exact


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
        NO_OF_ELEMS = NO_OF_NODES[0] - 1
    elif NO_OF_DIMENSIONS == 2:
        mesh_gen = generate_2d_mesh
        NO_OF_ELEMS = (NO_OF_NODES[0] - 1) * (NO_OF_NODES[1] - 1)
    else:
        raise NotImplementedError('Number of spatial dimensions not supported.')

    """ Plotting parameters and visualisations """
    plt.rc('text', usetex=True)
    lw = 1.5
    ms = 5.
    x_left = DOMAIN_BOUNDARIES[0][0]
    x_right = DOMAIN_BOUNDARIES[1][0]

    """ Problem definition """
    source = source4
    exact = exact4
    nodes, IEN, ID, boundaries, lm, mat_dim = mesh_gen(NO_OF_NODES, DOMAIN_BOUNDARIES,
                                                       BOUNDARY_CONDITIONS_TYPE, BOUNDARY_CONDITIONS_VALUES)

    # print(f"Nodes:\n{nodes}")
    # print(f"IEN:\n{IEN}")
    # print(f"ID:\n{ID}")
    # print(f"boundaries:\n{boundaries}")
    # print(f"lm:\n{lm}")

    cg = Mesh(nodes, IEN, ID, boundaries, NO_OF_DIMENSIONS, ELEM_SHAPE)
    cg.construct_elements(NO_OF_ELEMS, NO_OF_VARIABLES, source, P1)

    lhs, rhs, u = solve(cg, lm, NO_OF_ELEMS, P1)
    # print(f"Solutions:\n{u}")
    # print(f"Global Laplacian matrix:\n{lhs}")
    # print(f"Global force vector:\n{rhs}")

    """ Analytical solution """
    x = np.linspace(DOMAIN_BOUNDARIES[0][0], DOMAIN_BOUNDARIES[1][0], num=101, endpoint=True)
    u_exact = exact(x)

    plt.plot(x, u_exact, 'k', label='Exact', lw=lw)
    plt.plot(nodes, u[:NO_OF_NODES[0]], 'bx--', label='Numerical', lw=lw, ms=ms)
    plt.legend(loc='best')
    plt.xlabel(r'$x$', usetex=True)
    plt.ylabel(r'$u$', usetex=True)
    plt.title(f'$N_e = ${NO_OF_ELEMS}')
    plt.grid()
    # plt.tick_params(axis='both', direction='in')
    plt.show()

    """ Calculate L2-norm error """
    # dx = cg.elements[0].calculate_elem_vol()
    #
    # # l2err = sp.linalg.norm(psi_exact - psi_A, 2) / sp.linalg.norm(psi_exact, 2)
    # l2err = compute_l2_err(u, u_exact, dx)
    # print(f"L2-norm error: {l2err}")
