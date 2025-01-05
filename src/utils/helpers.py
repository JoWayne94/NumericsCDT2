"""
Helper functions

Author: JWT
"""
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import os, sys

# Configure system path
path = os.path.dirname(__file__)

if (sys.platform[:3] == 'win') or (sys.platform[:3] == 'Win'):
    sys.path.append(os.path.abspath(os.path.join(path, '..\..')))
else:
    sys.path.append(os.path.abspath(os.path.join(path, '../..')))


def extend_lm(original_lm, n_elems, p):
    """
    @:brief Extend the original lm matrix by adding rows for higher-order elements.

    :param original_lm: The original lm matrix with shared interface DOFs
    :param n_elems:     Number of elements in the mesh
    :param p:           Polynomial degree of the basis functions

    :return: Extended lm matrix including all higher-order DOFs
    """
    if p < 2:
        return original_lm.copy()

    # Initialize a list to hold the extended rows
    new_rows = []

    # Start global DOF numbering from the next available index
    next_dof = original_lm.max() + 1

    # Add rows for higher-order internal DOFs (if p > 1)
    for i in range(p - 1):
        new_row = []
        for elem in range(n_elems):
            # Add two unique DOFs for each element (per quadrature node)
            new_row.append(next_dof)
            next_dof += 1
        new_rows.append(new_row)

    # Combine the original lm with the new rows
    extended_lm = np.vstack((original_lm, new_rows))

    return extended_lm

def enforce_boundary_conditions(mesh, e, n_nodes, lm, u, global_force, lhs, dirichlet_list, unique_nodes):

    for node in mesh.elements[e].geom_data.ien:
        if node in mesh.connectivity_data.boundaries:
            if mesh.connectivity_data.ids[node] == -1:
                dirichlet_list.append(node)

                for b in range(n_nodes):  # Loop over all local DOFs of the first element
                    B = lm[b, e]
                    if B >= 0:
                        # Modify force vector for Dirichlet BC
                        global_force[B] -= mesh.connectivity_data.boundaries[node] * lhs[0, b]  # [1, 0]

                u[node] = mesh.connectivity_data.boundaries[node]
            else:
                # Modify force vector for Neumann BC
                if node in unique_nodes:
                    global_force[mesh.connectivity_data.ids[node]] += mesh.connectivity_data.boundaries[node]


def mol(t, u, mass, lhs, force, t_func=lambda t: 1.):
    """
    @:brief Function to compute the time derivative of u via Method of Lines

    :param t:     Current time.
    :param u:     Current solution vector.
    :param mass:  Mass matrix (numpy array).
    :param lhs:   Global matrix from the LHS (numpy array).
    :param force:  Force vector (numpy array).
    :param t_func: Source term scaling as a function of time.

    :return:      Time derivative, du/dt.
    """
    # dudt = np.zeros_like(u)
    # Solve M dudt = f - K u
    # dudt = np.linalg.solve(mass, force - lhs @ u)
    dudt = sparse.linalg.spsolve(mass, t_func(t) * force - lhs @ u)

    return dudt


# Function to remove a row and a column in a sparse matrix
def delete_sparse_row_col(matrix, index):
    """
    @:brief Deletes the specified row and column from a sparse matrix.

    :param matrix: scipy.sparse.lil_matrix or other sparse matrix
    :param index:  Index of the row/column to remove
    :return:       Modified sparse matrix
    """
    # Convert to CSR format for efficient slicing
    matrix = matrix.tocsr()

    # Remove the row
    mask = np.ones(matrix.shape[0], dtype=bool)
    mask[index] = False
    matrix = matrix[mask, :]  # Keep only the rows we want

    # Remove the column
    matrix = matrix[:, mask]  # Keep only the columns we want

    # Return as LIL format (if needed for further modifications)
    return matrix  # matrix.tolil()

def compute_l2_err(numerical_solution, true_solution, d):

    if numerical_solution.shape != true_solution.shape:

        raise ValueError("Numerical and analytical solution dimensions do not match.")

    return np.sqrt(np.sum(d * np.power(numerical_solution - true_solution, 2))) / np.sqrt(np.sum(d * np.power(true_solution, 2)))

def get_tri_area_coordinates(p, alpha, l):

    n = np.ones(l.shape[0])

    n *= get_eta_function(p, alpha[0], l[:,0])
    n *= get_eta_function(p, alpha[1], l[:,1])
    n *= get_eta_function(p, alpha[2], l[:,2])

    return n

def get_eta_function(p, alpha, l, skip=-1):

    index = np.concatenate((np.arange(0, skip), np.arange(skip + 1, alpha)))

    eta = np.ones(l.shape[0])

    for i in index:
        eta *= (p * l - i) / (i + 1.)

    return eta

def get_grad_eta_function(p, alpha, l):

    get_a = np.zeros_like(l)

    for i in range(int(alpha)):
        get_a += (p / (i + 1)) * get_eta_function(p, alpha, l, i)

    return get_a

def get_tri_grad_area_coordinates(p, alpha, l):

    dn = np.ones((l.shape[0], 3))

    n1 = get_eta_function(p, alpha[0], l[:,0])
    n2 = get_eta_function(p, alpha[1], l[:,1])
    n3 = get_eta_function(p, alpha[2], l[:,2])

    dn1 = get_grad_eta_function(p, alpha[0], l[:,0])
    dn2 = get_grad_eta_function(p, alpha[1], l[:,1])
    dn3 = get_grad_eta_function(p, alpha[2], l[:,2])

    dn[:, 0] = dn1 * n2 * n3
    dn[:, 1] = n1 * dn2 * n3
    dn[:, 2] = n1 * n2 * dn3

    return dn

""" Plotting functions """
def mapping(n_e, ien, lm, solution, dest):
    """
    @:brief Map solution back for visualisation

    :param n_e:      Number of elements.
    :param ien:      Integer element node array.
    :param lm:       Location mapping.
    :param solution: Solution vector from ODE solver.
    :param dest:     Destination vector.
    """
    # for elem in range(n_e):
    #     for node in range(ien[0].shape[0]):
    #         if lm[node, elem] >= 0:
    #             dest[ien[elem, node]] = solution[lm[node, elem]]

    # Boolean mask for valid indices in lm
    valid_indices = lm.T >= 0  # lm[node, elem] -> lm.T[elem, node]

    # Assign directly to dest; # ien[elem, node] is valid for lm[node, elem]
    dest[ien[valid_indices]] = solution[lm.T[valid_indices]]

def plot_2d_quad_mesh(nodes, ien, lw):

    # Extract the x and y coordinates of the nodes
    x = nodes[:, 0]
    y = nodes[:, 1]

    # Plot the quadrilateral mesh
    plt.figure(figsize=(8, 8))
    for quad in ien:
        x_coords = [x[quad[0]], x[quad[1]], x[quad[3]], x[quad[2]], x[quad[0]]]
        y_coords = [y[quad[0]], y[quad[1]], y[quad[3]], y[quad[2]], y[quad[0]]]
        plt.plot(x_coords, y_coords, 'k-', lw=lw)

    plt.scatter(x, y, c='red', s=50, label='Nodes')  # Mark the nodes
    plt.xlabel("x-coordinate", usetex=True)
    plt.ylabel("y-coordinate", usetex=True)
    plt.title("2D Quadrilateral mesh", usetex=True)
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.show()

def plot_2d_tri_mesh(nodes, ien, lw):

    # Extract the x and y coordinates of the nodes
    x = nodes[:, 0]
    y = nodes[:, 1]

    # Using triplot for mesh visualization
    plt.figure(figsize=(8, 8))
    plt.triplot(x, y, ien, color='k', lw=lw)  # Plot the triangles
    plt.scatter(x, y, c='red', s=50, label='Nodes')  # Mark the nodes
    plt.xlabel("x-coordinate", usetex=True)
    plt.ylabel("y-coordinate", usetex=True)
    plt.title("2D Triangular mesh", usetex=True)
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.show()

def plot_2d(x, y, u, x_left, x_right, y_bottom, y_top, title,
            show=False, z_min=None, z_max=None):

    if z_min is None: z_min = np.min(u)
    if z_max is None: z_max = np.max(u)

    fig = plt.figure(figsize=(10, 5))

    # Subplot 1: imshow
    ax1 = fig.add_subplot(1, 2, 1)
    im = ax1.imshow(u, extent=[x_left, x_right, y_bottom, y_top], origin='lower', cmap='viridis', aspect='auto',
                    vmin=z_min, vmax=z_max)
    ax1.set_xlabel(r'$x$', usetex=True)
    ax1.set_ylabel(r'$y$', usetex=True)
    ax1.set_xlim([x_left, x_right])
    ax1.set_ylim([y_bottom, y_top])
    cbar = fig.colorbar(im, ax=ax1, label=r'$u$')  # Individual color bar for imshow

    # Subplot 2: plot_surface with adjustable camera angle
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    surf = ax2.plot_surface(x, y, u, cmap='Blues_r', edgecolor='none')

    # Adjust the camera angle (elevation, azimuth)
    ax2.view_init(elev=30, azim=45)  # Change these values to adjust the view
    ax2.set_xlabel(r'$x$', usetex=True)
    ax2.set_ylabel(r'$y$', usetex=True)
    ax2.set_zlabel(r'$u$', usetex=True)
    ax2.set_xlim([x_right, x_left])
    ax2.set_ylim([y_top, y_bottom])
    ax2.set_zlim([z_min, z_max])
    plt.suptitle(title, fontsize=16, usetex=True)

    # Show the plot
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if show: plt.show()
    else: plt.pause(0.01)


def plot_2d_transient(x, y, u, x_left, x_right, y_bottom, y_top, title, frame_dir, frame_files, n,
                      gif=False, z_min_g=None, z_max_g=None):

    # Replot
    plt.cla()
    plot_2d(x, y, u, x_left, x_right, y_bottom, y_top, title, False, z_min_g, z_max_g)

    # Save current frame as an image file
    if gif:
        frame_filename = os.path.join(frame_dir, f'frame_{n:03d}.png')
        plt.savefig(frame_filename)
        frame_files.append(frame_filename)


def plot_2d_solutions(u, u_exact, nx, ny, x_left, x_right, y_bottom, y_top):

    x = np.linspace(x_left, x_right, nx)
    y = np.linspace(y_bottom, y_top, ny)
    X, Y = np.meshgrid(x, y)
    U = u.reshape(ny, nx)
    U_exact = u_exact.reshape(ny, nx)

    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf1 = ax1.plot_surface(X, Y, U_exact, cmap='Blues_r', edgecolor='none')

    z_min = np.min(u_exact)
    z_max = np.max(u_exact)

    # Adjust the camera angle (elevation, azimuth)
    ax1.view_init(elev=30, azim=45)  # Change these values to adjust the view
    ax1.set_xlabel(r'$x$', usetex=True)
    ax1.set_ylabel(r'$y$', usetex=True)
    ax1.set_zlabel(r'$u$', usetex=True)
    ax1.set_xlim([x_right, x_left])
    ax1.set_ylim([y_top, y_bottom])
    ax1.set_xlim([x_right, x_left])
    ax1.set_zlim([z_min, z_max])
    ax1.set_title(r'Exact solution', fontsize=16, usetex=True)

    # Subplot 2: plot_surface with adjustable camera angle
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    surf2 = ax2.plot_surface(X, Y, U, cmap='Blues_r', edgecolor='none')

    z_min = np.min(u)
    z_max = np.max(u)

    # Adjust the camera angle (elevation, azimuth)
    ax2.view_init(elev=30, azim=45)  # Change these values to adjust the view
    ax2.set_xlabel(r'$x$', usetex=True)
    ax2.set_ylabel(r'$y$', usetex=True)
    ax2.set_zlabel(r'$u$', usetex=True)
    ax2.set_xlim([x_right, x_left])
    ax2.set_ylim([y_top, y_bottom])
    ax2.set_zlim([z_min, z_max])
    ax2.set_title(r'Numerical solution', fontsize=16, usetex=True)
    # plt.suptitle(r'Numerical solution', fontsize=16, usetex=True)

    # Show the plot
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


""" Define functions specific to case in study """
def plot_uk_map(nodes, boundary_nodes, towns, town_names, m_styles, ms, lw, show=False):

    # Extract the x and y coordinates of the nodes
    x = nodes[:, 0] / 10000000.
    y = nodes[:, 1] / 10000000.

    towns_x = towns[:, 0] / 10000000.
    towns_y = towns[:, 1] / 10000000.

    # Using triplot for mesh visualization
    plt.figure(figsize=(8, 8))
    for t in range(len(town_names)):
        plt.plot(towns_x[t], towns_y[t], m_styles[t], lw=lw, ms=ms, label=town_names[t])
    plt.plot(x[boundary_nodes], y[boundary_nodes], 'k-', lw=lw)
    plt.xlabel(r"x-coordinate [$10^6$ m]", usetex=True)
    plt.ylabel(r"y-coordinate [$10^6$ m]", usetex=True)
    plt.title("Towns/cities with historical wind data", usetex=True)
    plt.legend(loc='upper left')
    plt.axis("equal")
    plt.grid(True)

    # plt.savefig(f'{sys.path[-1]}/report/figures/uk_map.eps', dpi=1000)

    if show: plt.show()
    else: plt.pause(0.1)

def plot_uk_mesh(nodes, boundary_nodes, ien, lw, show=False):

    # Extract the x and y coordinates of the nodes
    x = nodes[:, 0] / 10000000.
    y = nodes[:, 1] / 10000000.

    # Using triplot for mesh visualization
    plt.figure(figsize=(8, 8))
    plt.triplot(x, y, ien, color='k', lw=lw)  # Plot the triangles
    plt.scatter(x[boundary_nodes], y[boundary_nodes], c='red', s=50, label='Dirichlet nodes')  # Mark the nodes
    plt.xlabel(r"x-coordinate [$10^6$ m]", usetex=True)
    plt.ylabel(r"y-coordinate [$10^6$ m]", usetex=True)
    plt.title("2D Triangular mesh", usetex=True)
    plt.legend()
    plt.axis("equal")
    plt.grid(True)

    # plt.savefig(f'{sys.path[-1]}/report/figures/uk_mesh.eps', dpi=1000)

    if show: plt.show()
    else: plt.pause(0.1)

def plot_uk_solutions(x, y, u, ien, x_left, x_right, y_bottom, y_top, title, ms,
                      soton=None, reading=None, z_min=None, z_max=None,
                      vel_x=None, vel_y=None, r_e=None, s_e=None, show=True):

    if z_min is None: z_min = np.min(u)
    if z_max is None: z_max = np.max(u)

    # x /= 10000000.
    # y /= 10000000.
    # x_left /= 10000000.
    # x_right /= 10000000.
    # y_bottom /= 10000000.
    # y_top /= 10000000.
    # if soton is not None: soton /= 10000000.
    # if reading is not None: reading /= 10000000.

    fig = plt.figure(figsize=(6, 5)) # 10

    # Subplot 1: imshow
    ax1 = fig.add_subplot(1, 1, 1) # ax1 = fig.add_subplot(1, 2, 1)
    trip = ax1.tripcolor(x / 10000000., y / 10000000., u, triangles=ien, cmap='viridis', vmin=z_min, vmax=z_max)
    if r_e is not None: ax1.plot(np.append(x[ien[r_e]][0], x[ien[r_e]][0][0]) / 10000000.,
                                np.append(y[ien[r_e]][0], y[ien[r_e]][0][0]) / 10000000., 'k-', lw=1.5)
    if s_e is not None: ax1.plot(np.append(x[ien[s_e]][0], x[ien[s_e]][0][0]) / 10000000.,
                                np.append(y[ien[s_e]][0], y[ien[s_e]][0][0]) / 10000000., 'k-', lw=1.5)
    ax1.plot([soton[0] / 10000000.], [soton[1] / 10000000.], 'rx', ms=ms)
    ax1.plot([reading[0] / 10000000.], [reading[1] / 10000000.], 'ro', ms=ms)
    if vel_x is not None and vel_y is not None:
        ax1.quiver(x[::2] / 10000000., y[::2] / 10000000., vel_x[::2] / 10000000., vel_y[::2] / 10000000., angles='xy', scale_units='xy', scale=0.001, color='k')
    ax1.set_xlabel(r'$x$ [$10^6$ m]', usetex=True)
    ax1.set_ylabel(r'$y$ [$10^6$ m]', usetex=True)
    ax1.set_xlim([x_left / 10000000., x_right / 10000000.])
    ax1.set_ylim([y_bottom / 10000000., y_top / 10000000.])
    cbar = fig.colorbar(trip, ax=ax1, label=r'$u$')

    # Subplot 2: plot_surface with adjustable camera angle
    # ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    # surf = ax2.plot_trisurf(x, y, u, triangles=ien, cmap='Blues_r', edgecolor='none')
    #
    # # Adjust the camera angle (elevation, azimuth)
    # ax2.view_init(elev=30, azim=45)  # Change these values to adjust the view
    # ax2.set_xlabel(r'$x$ [m]', usetex=True)
    # ax2.set_ylabel(r'$y$ [m]', usetex=True) # [$10^6$ m]
    # ax2.set_zlabel(r'$u$', usetex=True)
    # ax2.set_xlim([x_right, x_left])
    # ax2.set_ylim([y_top, y_bottom])
    # ax2.set_zlim([z_min, z_max])
    plt.suptitle(title, fontsize=16, usetex=True)

    # plt.savefig(f'{sys.path[-1]}/report/figures/{title}.eps', dpi=1000)

    # Show the plot
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if show:
        plt.show()
    else:
        plt.pause(0.01)

def plot_uk_transient(x, y, u, ien, x_left, x_right, y_bottom, y_top, title, ms,
                      frame_dir, frame_files, n,
                      gif=False, soton=None, reading=None, z_min=None, z_max=None,
                      vel_x=None, vel_y=None):

    # Replot
    plt.cla()
    plot_uk_solutions(x, y, u, ien, x_left, x_right, y_bottom, y_top, title, ms,
                      soton, reading, z_min, z_max, vel_x, vel_y, show=False)

    # Save current frame as an image file
    if gif:
        frame_filename = os.path.join(frame_dir, f'frame_{n:03d}.png')
        plt.savefig(frame_filename)
        frame_files.append(frame_filename)

def plot_vel_field(nodes, boundary_nodes, towns, town_names, u, v, title, m_styles, ms, lw, show=False):

    # Extract the x and y coordinates of the nodes
    x = nodes[:, 0] / 10000000.
    y = nodes[:, 1] / 10000000.

    towns_x = towns[:, 0] / 10000000.
    towns_y = towns[:, 1] / 10000000.

    # Plot quiver plot
    plt.figure(figsize=(8, 8))
    for t in range(len(town_names)):
        plt.plot(towns_x[t], towns_y[t], m_styles[t], lw=lw, ms=ms, label=town_names[t])
    plt.plot(x[boundary_nodes], y[boundary_nodes], 'k-', lw=lw)

    plt.quiver(x[::4], y[::4], u[::4], v[::4], angles='xy', scale_units='xy', scale=10000, color='k')
    plt.title(title, usetex=True)
    plt.xlabel(r"x-coordinate [$10^6$ m]", usetex=True)
    plt.ylabel(r"y-coordinate [$10^6$ m]", usetex=True)
    # plt.legend(loc='upper left')
    plt.axis("equal")
    plt.grid(True)

    # plt.savefig(f'{sys.path[-1]}/report/figures/{title}.eps', dpi=1000)

    if show:
        plt.show()
    else:
        plt.pause(0.1)

def plot_vel_transient(nodes, boundary_nodes, towns, town_names, u, v, title, m_styles, ms, lw,
                      frame_dir, frame_files, n, gif=False):

    # Replot
    plt.cla()
    plot_vel_field(nodes, boundary_nodes, towns, town_names, u, v, title, m_styles, ms, lw, False)

    # Save current frame as an image file
    if gif:
        frame_filename = os.path.join(frame_dir, f'frame_{n:03d}.png')
        plt.savefig(frame_filename)
        frame_files.append(frame_filename)
