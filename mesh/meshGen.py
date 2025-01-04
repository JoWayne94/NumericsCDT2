"""
Methods to generate a mesh with either quadrilateral or triangle elements.

Notes:

    1. Also generates element mappings and BCs values

Author: JWT
"""
import numpy as np


def generate_1d_mesh(nn, db, bct, bcv):
    """
    @:brief Generate a one-dimensional mesh

    :param nn:            Number of nodes
    :param db:            Domain boundaries coordinates
    :param bct:           Boundary conditions' type
    :param bcv:           Boundary conditions' values

    :return: nodes coordinates, integer element node array, integer destination array
    """

    nx = nn[0] - 1
    nodes = np.linspace(db[0][0], db[1][0], num=nx + 1, endpoint=True)
    # nodes = np.array([db[0][0], *sorted(np.random.rand(nx)), db[1][0]])
    ids = np.zeros(len(nodes), dtype=np.int64)
    boundaries = dict()  # hold the boundary values
    n_eq = 0

    """ Directly construct location matrix """
    # LM = np.zeros((2, nx), dtype=np.int64)
    # for e in range(nx):
    #     if e == 0:
    #         LM[0, e] = -1
    #         LM[1, e] = 0
    #     else:
    #         LM[0, e] = LM[1, e - 1]
    #         LM[1, e] = LM[0, e] + 1

    """ Construct IEN and ID arrays """
    for nID, node in enumerate(nodes):
        is_boundary = False

        for i, boundary in enumerate(db):  # Iterate over boundary conditions (left and right)
            if np.allclose(node, boundary[0]):  # Check if the node is on the boundary
                is_boundary = True

                if bct[i] == 'Dirichlet':
                    ids[nID] = -1
                    # n_eq += 1
                    boundaries[nID] = bcv[i]

                elif bct[i] == 'Neumann':
                    ids[nID] = n_eq
                    n_eq += 1
                    boundaries[nID] = bcv[i]

                elif bct[i] == 'Periodic':
                    if i == 0:
                        ids[nID] = n_eq
                        n_eq += 1
                    else:
                        ids[nID] = ids[0]
                    for n in range(nodes.shape[0]):
                        if np.allclose(nodes[n], bcv[i]):
                            boundaries[nID] = n
                            break

                else:
                    raise NotImplementedError('BC type not implemented.')
                break  # Exit the loop once boundary condition is applied

        if not is_boundary:  # If the node is not on any boundary
            ids[nID] = n_eq
            n_eq += 1

    ien = np.zeros((nx, 2), dtype=np.int64)
    for i in range(nx):
        ien[i, :] = (i, i + 1)

    lm = np.zeros_like(ien.T)
    for e in range(ien.shape[0]):
        for a in range(ien.shape[1]):
            lm[a, e] = ids[ien[e, a]]

    return nodes, ien, ids, boundaries, lm, n_eq


def generate_2d_mesh(nn, db, bct, bcv, shape):
    """
    @:brief Generate a two-dimensional mesh in a rectangular domain

    :param nn:            Number of nodes
    :param db:            Domain boundaries coordinates
    :param bct:           Boundary conditions' type
    :param bcv:           Boundary conditions' values
    :param shape:         Shape of the elements

    :return: nodes coordinates, integer element node array, integer destination array
    """
    nx = nn[0]
    ny = nn[1]
    x = np.linspace(db[0][0], db[1][0], nx)
    y = np.linspace(db[0][1], db[2][1], ny)
    X, Y = np.meshgrid(x, y)
    nodes = np.zeros((nx * ny, 2))
    nodes[:, 0] = X.ravel()
    nodes[:, 1] = Y.ravel()
    ids = np.zeros(len(nodes), dtype=np.int64)
    boundaries = dict()  # hold the boundary values
    n_eq = 0

    """ Construct IEN and ID arrays """
    for nID, node in enumerate(nodes):
        is_boundary = False

        for i, boundary in enumerate(db):  # Iterate over boundary conditions (left, bottom, right, top)
            if np.allclose(node[i % 2], boundary[i % 2]):  # Check if the node is on the boundary
                is_boundary = True
                if bct[i] == 'Dirichlet':
                    ids[nID] = -1
                    # n_eq += 1
                    boundaries[nID] = bcv[i]  # Dirichlet BC
                    break
                elif bct[i] == 'Neumann' and ids[nID] == 0:
                    # ids[nID] = n_eq
                    # n_eq += 1
                    boundaries[nID] = bcv[i]  # Neumann BC
                elif bct[i] == 'Periodic':
                    if i == 2 or i == 3:
                        n_eq -= 1
                else:
                    raise NotImplementedError('BC type not implemented.')
                # break  # Exit the loop once boundary condition is applied

        if not is_boundary or ids[nID] == 0:  # If the node is not on any boundary
            ids[nID] = n_eq
            n_eq += 1

    """ Periodic treated after """
    if bct[0] == 'Periodic' and bct[2] == 'Periodic':
        for row in range(ny):
            ids[row * nx + (nx - 1)] = ids[row * nx]
            if str(row * nx) in boundaries:
                boundaries[str(row * nx) + '_2'] = row * nx + (nx - 1)
            else:
                boundaries[str(row * nx)] = row * nx + (nx - 1)

    if bct[1] == 'Periodic' and bct[3] == 'Periodic':
        for col in range(nx):
            ids[(ny - 1) * nx + col] = ids[col]
            if str(col) in boundaries:
                boundaries[str(col) + '_2'] = (ny - 1) * nx + col
            else:
                boundaries[str(col)] = (ny - 1) * nx + col

    ne_x = nx - 1
    ne_y = ny - 1

    if shape == 'Q':
        ien = np.zeros((ne_x * ne_y, 4), dtype=np.int64)
        for i in range(ne_x):
            for j in range(ne_y):
                ien[i + j * ne_x, :] = (i + j * nx, i + 1 + j * nx, i + (j + 1) * nx, i + 1 + (j + 1) * nx)
    elif shape == 'T':
        ien = np.zeros((2 * ne_x * ne_y, 3), dtype=np.int64)
        for i in range(ne_x):
            for j in range(ne_y):
                ien[2 * i + 2 * j * ne_x, :] = (i + j * nx, i + 1 + j * nx, i + (j + 1) * nx)
                ien[2 * i + 1 + 2 * j * ne_x, :] = (i + 1 + j * nx, i + 1 + (j + 1) * nx, i + (j + 1) * nx)
    else:
        raise NotImplementedError('Shape type not implemented.')

    lm = np.zeros_like(ien.T)
    for e in range(ien.shape[0]):
        for a in range(ien.shape[1]):
            lm[a, e] = ids[ien[e, a]]

    return nodes, ien, ids, boundaries, lm, n_eq
