"""
2d unsteady ADR equation on a UK grid

Author: JWT
"""

# Import third-party libraries
import sys, os
import imageio
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

# Import equation types
from src.solvers.Helmholtz import *
from src.solvers.ADR import *


def source1(coords):
    return 1.

def source2(coords, std=1000.):
    return np.exp(-1. / (2. * std ** 2) * ((coords[0] - soton[0]) ** 2 + (coords[1] - soton[1]) ** 2))

def source_t(time, amp=1., mid=0.5 * SOTON_FIRE, spread=8. / 3.):
    std = mid / spread
    return amp * np.exp(-1. / (2. * std ** 2) * ((time - mid) ** 2))


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

        dirichlet_boundary = np.where(nodes[boundary_nodes, 1] <= 150000.)[0]

        # Make all boundary points Dirichlet
        ID = np.zeros(len(nodes), dtype=np.int64)
        boundaries = dict()  # hold the boundary values
        n_eq = 0
        for i in range(len(nodes[:, 1])):
            if i in dirichlet_boundary:
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

        # plot_uk_mesh(nodes, dirichlet_boundary, IEN, lw)

    else:
        nodes, IEN, ID, boundaries, lm, mat_dim = mesh_gen(NO_OF_NODES, DOMAIN_BOUNDARIES,
                                                           BOUNDARY_CONDITIONS_TYPE,
                                                           BOUNDARY_CONDITIONS_VALUES, ELEM_SHAPE)

    # print(f"nodes:\n{nodes}")
    # print(f"IEN:\n{IEN}")
    # print(f"ID:\n{ID}")
    # print(f"boundaries:\n{boundaries}")
    # print(f"lm:\n{lm}")

    directed_at_reading = reading - soton
    VEL_FIELD = -10. / np.linalg.norm(directed_at_reading) * directed_at_reading

    """ Mesh construction """
    cg = Mesh(nodes, IEN, ID, boundaries, NO_OF_DIMENSIONS, ELEM_SHAPE)
    cg.construct_elements(NO_OF_ELEMS, NO_OF_VARIABLES, source, P1, P2)

    lap, stn, force, mass, u, mask = solve(cg, lm, NO_OF_ELEMS, P1, P2, VEL_FIELD, KAPPA)
    # print(f"Solutions:\n{u}")
    # print(f"Global Laplacian matrix:\n{lhs}")
    # print(f"Global force vector:\n{rhs}")

    """ Initial conditions """
    # Initial datum/conditions defined with a numpy array
    u0 = np.zeros_like(u)

    """ Visualisation """
    x_left = np.min(nodes[:, 0])
    x_right = np.max(nodes[:, 0])
    y_bottom = np.min(nodes[:, 1])
    y_top = np.max(nodes[:, 1])
    # Directory to save frames
    frame_dir = 'frames'
    os.makedirs(frame_dir, exist_ok=True)
    # Initialize empty list to store frame filenames
    frame_files = []

    """ Time-step size """
    h = abs(nodes[1, 0] - nodes[0, 0])  # very rough estimate
    mag_v = np.sqrt(VEL_FIELD[0] ** 2 + VEL_FIELD[1] ** 2)
    dt_a = CFL * h / mag_v
    dt_d = (h ** 2) / (4. * np.abs(KAPPA))
    print(f"Time-step size limited by advection: {dt_a}")
    print(f"Time-step size due to diffusion restriction: {dt_d}")
    max_dt = 0.2 * min(dt_a, dt_d)

    t = 0.
    n = 0
    create_gif = False
    # Set up the ODE solver
    u = np.empty_like(u0)
    r = ode(mol).set_integrator('dopri5', max_step=max_dt)
    r.set_initial_value(u0[mask], t)
    r.set_f_params(mass, lap + stn, force, source_t)  # Pass additional arguments to dudt

    dt = 900.
    print(f"Number of time intervals: {int(FINAL_TIME / dt)}.")

    reading_elem = np.array([0], dtype=np.int32)
    soton_elem = np.array([0], dtype=np.int32)
    reading_local_coords = np.zeros((1, 2), dtype=np.float64)
    soton_local_coords = np.zeros((1, 2), dtype=np.float64)
    elem_indicator = np.array([0, 0], dtype=np.int32)
    for elem in range(NO_OF_ELEMS):
        reading_local_coords = cg.elements[elem].geoms.geometry.inv_mapping(reading)
        soton_local_coords = cg.elements[elem].geoms.geometry.inv_mapping(soton)

        if reading_local_coords[0][0] >= 0. and reading_local_coords[0][1] >= 0. and np.sum(
                reading_local_coords[0]) < np.sqrt(2.):
            reading_elem[0] = elem
            elem_indicator[0] = 1
        if soton_local_coords[0][0] >= 0. and soton_local_coords[0][1] >= 0. and np.sum(
                soton_local_coords[0]) < np.sqrt(2.):
            soton_elem[0] = elem
            elem_indicator[1] = 1

        if elem_indicator[0] == 1 and elem_indicator[1] == 1:
            break

    reading_values = []
    soton_values = []

    # Time-stepping
    while r.successful() and abs(FINAL_TIME - r.t) > 1.e-8:

        if r.t + dt >= FINAL_TIME: dt = FINAL_TIME - r.t

        r.integrate(r.t + dt)  # Integrate up to the next time step

        mapping(NO_OF_ELEMS, IEN, lm, r.y, u)

        # u /= np.max(u)

        reading_u = (u[IEN[reading_elem[0], 0]] * (1. - reading_local_coords[0][0] - reading_local_coords[0][1]) +
                     u[IEN[reading_elem[0], 1]] * reading_local_coords[0][0] +
                     u[IEN[reading_elem[0], 2]] * reading_local_coords[0][1])

        soton_u = (u[IEN[soton_elem[0], 0]] * (1. - reading_local_coords[0][0] - reading_local_coords[0][1]) +
                   u[IEN[soton_elem[0], 1]] * reading_local_coords[0][0] +
                   u[IEN[soton_elem[0], 2]] * reading_local_coords[0][1])

        reading_values.append(reading_u)
        soton_values.append(soton_u)

        # plot_uk_transient(nodes[:, 0], nodes[:, 1], u, IEN, x_left, x_right, y_bottom, y_top,
        #                   r'Time = ' + '%.2f' % r.t + ' s', ms,
        #                   frame_dir, frame_files, n, create_gif, soton, reading)

        n += 1

    # Create the GIF
    if create_gif:
        with imageio.get_writer(f'las_{INPUT_NAME}k.gif', mode='I', duration=0.004) as writer:
            for filename in frame_files:
                image = imageio.imread(filename)
                writer.append_data(image)

        # Optional: Clean up the frame files after GIF creation
        for filename in frame_files:
            os.remove(filename)

    # r.integrate(FINAL_TIME)
    print(f"Final time of {r.t} seconds reached.")
    print(f"Pollutant concentration values at reading:\n{reading_values}.")
    print(f"Pollutant concentration values at soton:\n{soton_values}.")

    mapping(NO_OF_ELEMS, IEN, lm, r.y, u)

    plt.close('all')
    plot_uk_solutions(nodes[:, 0], nodes[:, 1], u, IEN, x_left, x_right, y_bottom, y_top,
                    f"$N_e = {NO_OF_ELEMS}$, unsteady advection-diffusion case", ms, soton, reading,
                      r_e=reading_elem, s_e=soton_elem)
