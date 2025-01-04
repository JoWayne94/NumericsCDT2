"""
1D Poisson convergence test by varying dx

Author: JWT
"""

# Import third-party libraries
import sys, os
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations

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

def compute_l2_err(numerical_solution, true_solution, dx):

    return np.sqrt(np.sum(dx * np.power(numerical_solution - true_solution, 2))) / np.sqrt(np.sum(dx * np.power(true_solution, 2)))


if __name__ == '__main__':
    """
    main()
    """

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
    line_colours = ['rx--', 'go--', 'bs--', 'y', 'k']

    x_left = DOMAIN_BOUNDARIES[0][0]
    x_right = DOMAIN_BOUNDARIES[1][0]

    """ Problem definition """
    source = source4
    exact = exact4

    """ Define number of nodes to be tested """
    nnodes_list = [9, 17, 33, 65, 129]

    """ Error measures """
    L2_errors = np.empty(len(nnodes_list))

    for nn in range(len(nnodes_list)):

        NO_OF_ELEMS = nnodes_list[nn] - 1

        """ Mesh construction """
        nodes, IEN, ID, boundaries, lm, mat_dim = mesh_gen([nnodes_list[nn]], DOMAIN_BOUNDARIES,
                                                           BOUNDARY_CONDITIONS_TYPE, BOUNDARY_CONDITIONS_VALUES)

        cg = Mesh(nodes, IEN, ID, boundaries, NO_OF_DIMENSIONS, ELEM_SHAPE)
        cg.construct_elements(NO_OF_ELEMS, NO_OF_VARIABLES, source, P1)

        """ Solve """
        lhs, rhs, u = solve(cg, lm, NO_OF_ELEMS, P1)

        """ Analytical solution """
        u_exact = exact(nodes)

        """ Append L2 error to list """
        L2_errors[nn] = compute_l2_err(u, u_exact, cg.elements[0].calculate_elem_vol())


    nnodes_list = [9., 17., 33., 65., 129.]
    dx_list = [(DOMAIN_BOUNDARIES[1][0] - DOMAIN_BOUNDARIES[0][0]) / d for d in np.array(nnodes_list) - 1]

    """ Compute numerical order of convergence """
    # Calculate n for each unique pair of (epsilon, delta x) values
    n_values = []
    for (i, j) in combinations(range(len(L2_errors)), 2):
        epsilon_1, epsilon_2 = L2_errors[i], L2_errors[j]
        delta_x_1, delta_x_2 = dx_list[i], dx_list[j]

        # Using the formula provided to calculate n for each pair
        n = (np.log(epsilon_1) - np.log(epsilon_2)) / (np.log(delta_x_1) - np.log(delta_x_2))
        n_values.append(n)

    # Calculate the average n
    average_n = np.mean(n_values)

    # Output the result
    print(f"Calculated n values for each pair: {n_values}.")
    print(f"Average value of n {average_n}.")

    """ Plot convergence """
    plt.xscale('log')
    plt.yscale('log')

    plt.plot(np.array(nnodes_list) - 1, L2_errors, line_colours[0],
             label='test', lw=lw, ms=ms, fillstyle='none')

    # Define the desired gradient and constant
    m1 = -1  # Desired gradient (slope) on the log-log scale
    m2 = -2
    C1 = 7  # Constant that sets the vertical position of the line
    C2 = 3

    # Calculate y based on the power law y = C * x^m
    y1 = C1 * (np.array(nnodes_list) - 1) ** m1
    y2 = C2 * (np.array(nnodes_list) - 1) ** m2

    plt.plot(np.array(nnodes_list) - 1, y1, 'k:', label=r'$A \Delta x$', lw=lw, ms=ms)

    plt.legend(loc='best')
    plt.xlabel(r'$\Delta x$', usetex=True)
    # plt.xticks(np.array(nnodes_list) - 1, [f"$2 \pi / {int(nx)}$" for nx in np.array(nnodes_list) - 1])
    plt.ylabel(r'$L_2$ error', usetex=True)
    plt.title(r'Convergence on a log$_{10}$-log$_{10}$ scale', usetex=True)
    plt.grid()
    # plt.tick_params(axis='both', direction='in')
    # plt.savefig(f'{path}/convergence.eps', dpi=1000)
    plt.show()
