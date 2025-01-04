"""
Test cases for the 2d ADR equation

Author: JWT
"""

# Import third-party libraries
import sys, os
import imageio
import matplotlib.pyplot as plt
from scipy.integrate import ode


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
from src.solvers.ADR import *


def source1(coords):
    return 0.

def exact1(coords):
    return (np.sin(2. * np.pi * (coords[:, 0] - VEL_FIELD[0] * FINAL_TIME))
            * np.sin(2. * np.pi * (coords[:, 1] - VEL_FIELD[1] * FINAL_TIME))
            * np.exp(-8. * np.pi * np.pi * KAPPA * FINAL_TIME))

def mapping(solution, dest):
    """
    @:brief Map solution back for visualisation

    :param solution: Solution vector from ODE solver
    :param dest:     Destination vector
    """
    for e in range(NO_OF_ELEMS):
        for a in range(IEN[0].shape[0]):
            dest[IEN[e, a]] = solution[lm[a, e]]


if __name__ == '__main__':
    """
    main()
    """

    # if len(sys.argv) < 2:
    if EQUATION_TYPE == 'Helmholtz':
        solve = helmholtz_operator
    elif EQUATION_TYPE == 'ADR':
        solve = adr
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
    source = source1
    exact = exact1
    nodes, IEN, ID, boundaries, lm, mat_dim = mesh_gen(NO_OF_NODES, DOMAIN_BOUNDARIES,
                                                       BOUNDARY_CONDITIONS_TYPE, BOUNDARY_CONDITIONS_VALUES, ELEM_SHAPE)

    # print(f"nodes:\n{nodes}")
    # print(f"IEN:\n{IEN}")
    # print(f"ID:\n{ID}")
    # print(f"boundaries:\n{boundaries}")
    # print(f"lm:\n{lm}")

    """ Mesh construction """
    cg = Mesh(nodes, IEN, ID, boundaries, NO_OF_DIMENSIONS, ELEM_SHAPE)
    cg.construct_elements(NO_OF_ELEMS, NO_OF_VARIABLES, source, P1, P2)

    lhs, force, mass, u0, mask = solve(cg, lm, NO_OF_ELEMS, P1, P2, VEL_FIELD, KAPPA)
    # print(f"Solutions:\n{u}")
    # print(f"Global Laplacian matrix:\n{lhs}")
    # print(f"Global force vector:\n{rhs}")

    """ Initial conditions """
    # Initial datum/conditions defined with a numpy array
    for n in range(nodes.shape[0]):
        u0[n] = np.sin(2. * np.pi * nodes[n, 0]) * np.sin(2. * np.pi * nodes[n, 1])

    # mks?
    """ Plot ICs """
    x = np.linspace(x_left, x_right, NO_OF_NODES[0])
    y = np.linspace(y_bottom, y_top, NO_OF_NODES[1])
    X, Y = np.meshgrid(x, y)
    U = u0.reshape(NO_OF_NODES[1], NO_OF_NODES[0])
    # u = U.reshape(-1, )

    plot_2d(X, Y, U, x_left, x_right, y_bottom, y_top, r'Initial datum', show=True)
    z_max_global = np.max(u0)
    z_min_global = np.min(u0)

    # Directory to save frames
    frame_dir = 'frames'
    os.makedirs(frame_dir, exist_ok=True)
    # Initialize empty list to store frame filenames
    frame_files = []

    """ Time-step size """
    h = nodes[1, 0] - nodes[0, 0] # assume dx for now
    mag_v = np.sqrt(VEL_FIELD[0] ** 2 + VEL_FIELD[1] ** 2)
    # (CFL / (2. * P1 + 1))
    dt_a = CFL * h / mag_v
    dt_d = (h ** 2) / (4. * np.abs(KAPPA))
    print(f"Time-step size limited by advection: {dt_a}")
    print(f"Time-step size due to diffusion restriction: {dt_d}")
    dt = min(dt_a, dt_d)

    t = 0.
    n = 0
    create_gif = True
    # Set up the ODE solver
    u = np.empty_like(u0)
    r = ode(mol).set_integrator('dopri5', max_step=dt)
    r.set_initial_value(u0[mask], t)
    r.set_f_params(mass, lhs, force)  # Pass additional arguments to dudt

    # Time-stepping
    while r.successful() and abs(FINAL_TIME - r.t) > 1.e-8:

        if r.t + dt >= FINAL_TIME: dt = FINAL_TIME - r.t

        r.integrate(r.t + dt)  # Integrate up to the next time step

        mapping(r.y, u)

        plot_2d_transient(X, Y, u.reshape(NO_OF_NODES[1], NO_OF_NODES[0]), x_left, x_right, y_bottom, y_top,
                          r'CFL $= $' + str(CFL) + ', Time = ' + '%.2f' % (r.t + dt) + ' s',
                          frame_dir, frame_files, n, create_gif)

        n += 1

    # Create the GIF
    if create_gif:
        with imageio.get_writer('ADR.gif', mode='I', duration=0.002) as writer:
            for filename in frame_files:
                image = imageio.imread(filename)
                writer.append_data(image)

        # Optional: Clean up the frame files after GIF creation
        for filename in frame_files:
            os.remove(filename)

    # r.integrate(FINAL_TIME)
    # print('Time = ' + '%.5f' % r.t + ' s')
    print(f"Final time of {r.t} seconds reached.")
    mapping(r.y, u)

    """ Analytical solution """
    u_exact = exact(nodes)

    """ Plot solutions """
    # plt.cla()
    # plt.clf()
    plt.close('all')
    plot_2d_solutions(u, u_exact, NO_OF_NODES[0], NO_OF_NODES[1], x_left, x_right, y_bottom, y_top)
