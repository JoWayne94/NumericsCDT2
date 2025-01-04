"""
File: Euler.py

Description: Solve Euler equations in 1D
"""
import numpy as np
import math
from src.library.dgMesh.dgMesh import *
from src.library.paramCells.basis import *
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator


def conservativeToPrimitive(u):
    """
    @:brief Conservative to primitive variables
    :param u: Conservative variables
    :return: Primitive variables
    """
    w = np.empty(3)
    rho = u[0]
    vx = u[1] / u[0]
    E = u[2]
    kin = 0.5 * rho * vx * vx
    e = (E - kin) / rho
    gamma = 1.4
    pressure = rho * e * (gamma - 1.0)

    w[0] = rho
    w[1] = vx
    w[2] = pressure

    return w


def primitiveToConservative(w):
    """
    @:brief Primitive to conservative variables
    :param w: Primitive variables
    :return: Conservative variables
    """
    u = np.empty(3)
    rho = w[0]
    vx = w[1]
    pressure = w[2]
    kin = 0.5 * rho * vx * vx
    gamma = 1.4
    e = pressure / (rho * (gamma - 1.0))
    E = rho * e + kin

    u[0] = rho
    u[1] = rho * vx
    u[2] = E

    return u


def eulerFlux(u):
    """
    @:brief Compute the Euler flux from conservative variables
    :param u: Conservative variables
    :return: Euler flux
    """
    F = np.empty(3)
    w = conservativeToPrimitive(u)
    rho = w[0]  # abs(w[0])
    vx = w[1]
    pressure = w[2]  # abs(w[2])
    E = u[2]  # abs(u[2])

    F[0] = rho * vx
    F[1] = rho * vx * vx + pressure
    F[2] = vx * (E + pressure)

    return F


def computeSoundSpeed(rho, pressure):
    """
    @:brief Calculate speed of sound using Ideal Gas Law
    :param rho:      Density
    :param pressure: Pressure
    :return: Sound speed
    """
    # Ideal Gas EoS
    # a = math.sqrt(abs(1.4 * pressure / rho))
    a = math.sqrt(1.4 * pressure / rho)

    return a


def waveEstimates(ul, ur):
    """
    @:brief Compute left and right wave speeds using formulation from Toro's book
    :param ul: Left conservative variables of a face
    :param ur: Right conservative variables of a face
    :return: Left and right wave speeds
    """
    wl = conservativeToPrimitive(ul)
    wr = conservativeToPrimitive(ur)

    # rhol = abs(wl[0])
    # rhor = abs(wr[0])
    # vxl = wl[1]
    # vxr = wr[1]
    # pressurel = abs(wl[2])
    # pressurer = abs(wr[2])

    rhol = wl[0]
    rhor = wr[0]
    vxl = wl[1]
    vxr = wr[1]
    pressurel = wl[2]
    pressurer = wr[2]

    # Ideal Gas EoS
    al = computeSoundSpeed(rhol, pressurel)
    ar = computeSoundSpeed(rhor, pressurer)
    gamma = 1.4

    # El = abs(ul[2])
    # Er = abs(ur[2])
    #
    # """ Roe averaged state """
    # rhol_sqrt = math.sqrt(rhol)
    # rhor_sqrt = math.sqrt(rhor)
    # """ Total specific enthalpy """
    # Hl = (El + pressurel) / rhol
    # Hr = (Er + pressurer) / rhor
    #
    # vRoe = (rhol_sqrt * vxl + rhor_sqrt * vxr) / (rhol_sqrt + rhor_sqrt)
    # HRoe = (rhol_sqrt * Hl + rhor_sqrt * Hr) / (rhol_sqrt + rhor_sqrt)
    # c2Roe = abs((1.4 - 1) * (HRoe - 0.5 * vRoe * vRoe))
    # cRoe = math.sqrt(c2Roe)

    # /** Pressure–Based Wave Speed Estimates (ideal gases) **/
    """ Two-rarefaction Riemann solver TRRS for computing Pstar
        Toro's book page 301 & 330, section 9.4.1 """
    z = (gamma - 1) / (2.0 * gamma)
    pLR = (pressurel / pressurer) ** z
    vstar = (pLR * vxl / al + vxr / ar + 2.0 * (pLR - 1.0) / (gamma - 1.0)) / (pLR / al + 1.0 / ar)
    pstar = 0.5 * (pressurel * (1.0 + (gamma - 1.0) / (2.0 * al) * (vxl - vstar)) ** (1.0 / z) + pressurer
                   * (1.0 + (gamma - 1.0) / (2.0 * ar) * (vstar - vxr)) ** (1.0 / z))

    # /** Toro's book, page 330, eq (10.59 & 10.60) **/
    if pstar <= pressurel:
        ql = 1.0
    else:
        ql = math.sqrt(1.0 + (gamma + 1.0) / (2.0 * gamma) * (pstar / pressurel - 1.0))

    if pstar <= pressurer:
        qr = 1.0
    else:
        qr = math.sqrt(1.0 + (gamma + 1.0) / (2.0 * gamma) * (pstar / pressurer - 1.0))

    sleft = vxl - al * ql
    sright = vxr + ar * qr

    # sleft = vRoe - cRoe
    # sright = vRoe + cRoe

    return sleft, sright


def computeDt(leftfacearray, rightfacearray, dx, ncells, cfl):
    """
    @:brief Calculate time-step size with CFL constraints
    :param leftfacearray:  Conservative values on the left face of the cell
    :param rightfacearray: Conservative values on the right face of the cell
    :param dx:             Mesh size
    :param ncells:         Number of cells
    :param cfl:            Courant number
    :return: Time-step size
    """
    sleft, sright, smax = 0.0, 0.0, 0.0
    # Left and right conservative states of the RP

    for cell in range(ncells - 1):  # n cells + 1, meshObj.connectivityData.cells[i].solnCell.uPhysical
        ql = rightfacearray[cell]
        qr = leftfacearray[cell + 1]

        sleft, sright = waveEstimates(ql, qr)  # calculate left and right waves

        smax = max(smax, max(abs(sleft), abs(sright)))  # eq (6.19)

    return cfl * dx / smax  # // eq (6.17), make sure dt addition does not exceed final time output


def HLLFlux(qleft, qright):
    sleft, sright = waveEstimates(qleft, qright)  # calculate Sl and Sr

    fhll = np.empty(3)
    fl = eulerFlux(qleft)
    fr = eulerFlux(qright)

    # /** Eq (10.21) **/
    if sleft >= 0:
        return fl
    elif sright <= 0:
        return fr
    else:
        for n in range(3):
            fhll[n] = (sright * fl[n] - sleft * fr[n] + sleft * sright * (qright[n] - qleft[n])) / (sright - sleft)
        return fhll


def HLLCFlux(qleft, qright):
    wl = conservativeToPrimitive(qleft)
    wr = conservativeToPrimitive(qright)

    rhol = wl[0]
    rhor = wr[0]
    vxl = wl[1]
    vxr = wr[1]
    pressurel = wl[2]
    pressurer = wr[2]

    Sl, Sr = waveEstimates(qleft, qright)

    Fl = eulerFlux(qleft)
    Fr = eulerFlux(qright)

    Sstar = (pressurer - pressurel + rhol * vxl * (Sl - vxl) - rhor * vxr * (Sr - vxr)) / (
            rhol * (Sl - vxl) - rhor * (Sr - vxr))  # eq (10.37)

    # /** Variant 2 **/
    pLR = 0.5 * (pressurel + pressurer + rhol * (Sl - vxl) * (Sstar - vxl) + rhor * (Sr - vxr) * (
            Sstar - vxr))  # eq (10.42)
    Dstar = [0, 1, Sstar]  # 1D, eq (10.40)
    Qstarl = np.empty(3)
    Qstarr = np.empty(3)
    Fstarl = np.empty(3)
    Fstarr = np.empty(3)

    # /** Eq (10.43) **/
    for n in range(3):
        Qstarl[n] = (Sl * qleft[n] - Fl[n] + pLR * Dstar[n]) / (Sl - Sstar)
        Qstarr[n] = (Sr * qright[n] - Fr[n] + pLR * Dstar[n]) / (Sr - Sstar)

    # /** Eq (10.44) **/
    for n in range(3):
        Fstarl[n] = (Sstar * (Sl * qleft[n] - Fl[n]) + Sl * pLR * Dstar[n]) / (Sl - Sstar)
        Fstarr[n] = (Sstar * (Sr * qright[n] - Fr[n]) + Sr * pLR * Dstar[n]) / (Sr - Sstar)

    # /** Eq (10.26) **/
    if Sl >= 0:
        return Fl

    elif Sl <= 0 and Sstar >= 0:
        return Fstarl

    elif Sstar <= 0 and Sr >= 0:
        return Fstarr

    elif Sr <= 0:
        return Fr

    else:
        raise ValueError


def RoeSolver(qleft, qright):
    wl = conservativeToPrimitive(qleft)
    wr = conservativeToPrimitive(qright)

    Fl = eulerFlux(qleft)
    Fr = eulerFlux(qright)

    rhol = wl[0]
    rhor = wr[0]
    vxl = wl[1]
    vxr = wr[1]
    pressurel = wl[2]
    pressurer = wr[2]
    El = qleft[2]
    Er = qright[2]

    """ Roe averaged state """
    rhol_sqrt = math.sqrt(rhol)
    rhor_sqrt = math.sqrt(rhor)
    """ Total specific enthalpy """
    Hl = (El + pressurel) / rhol
    Hr = (Er + pressurer) / rhor

    vRoe = (rhol_sqrt * vxl + rhor_sqrt * vxr) / (rhol_sqrt + rhor_sqrt)
    HRoe = (rhol_sqrt * Hl + rhor_sqrt * Hr) / (rhol_sqrt + rhor_sqrt)
    rhoRoe = rhol_sqrt * rhor_sqrt
    c2Roe = (1.4 - 1) * (HRoe - 0.5 * vRoe * vRoe)
    cRoe = math.sqrt(c2Roe)

    drho = rhor - rhol
    dvx = vxr - vxl
    dpressure = pressurer - pressurel

    alpha = np.empty(3)
    alpha[0] = 0.5 / c2Roe * (dpressure - cRoe * rhoRoe * dvx)
    alpha[1] = drho - dpressure / c2Roe
    alpha[2] = 0.5 / c2Roe * (dpressure + cRoe * rhoRoe * dvx)

    Lambda = np.empty(3)
    Lambda[0] = vRoe - cRoe
    Lambda[1] = vRoe
    Lambda[2] = vRoe + cRoe

    R = np.empty((3, 3))
    R[0, :] = 1.0
    R[1, 0] = Lambda[0]
    R[1, 1] = Lambda[1]
    R[1, 2] = Lambda[2]
    R[2, 0] = HRoe - vRoe * cRoe
    R[2, 1] = 0.5 * vRoe * vRoe
    R[2, 2] = HRoe + vRoe * cRoe

    FRoe = np.einsum('kl, l -> k', R, np.abs(Lambda) * alpha)

    return 0.5 * (Fl + Fr - FRoe)


def RoeMatrix(qleft, qright):

    wl = conservativeToPrimitive(qleft)
    wr = conservativeToPrimitive(qright)

    rhol = wl[0]
    rhor = wr[0]
    vxl = wl[1]
    vxr = wr[1]
    pressurel = wl[2]
    pressurer = wr[2]
    El = qleft[2]
    Er = qright[2]

    """ Roe averaged state """
    rhol_sqrt = math.sqrt(rhol)
    rhor_sqrt = math.sqrt(rhor)
    """ Total specific enthalpy """
    Hl = (El + pressurel) / rhol
    Hr = (Er + pressurer) / rhor

    vRoe = (rhol_sqrt * vxl + rhor_sqrt * vxr) / (rhol_sqrt + rhor_sqrt)
    HRoe = (rhol_sqrt * Hl + rhor_sqrt * Hr) / (rhol_sqrt + rhor_sqrt)
    c2Roe = (1.4 - 1) * (HRoe - 0.5 * vRoe * vRoe)
    cRoe = math.sqrt(c2Roe)

    Lambda = np.empty(3)
    Lambda[0] = vRoe - cRoe
    Lambda[1] = vRoe
    Lambda[2] = vRoe + cRoe

    R = np.empty((3, 3))
    R[0, :] = 1.0
    R[1, 0] = Lambda[0]
    R[1, 1] = Lambda[1]
    R[1, 2] = Lambda[2]
    R[2, 0] = HRoe - vRoe * cRoe
    R[2, 1] = 0.5 * vRoe * vRoe
    R[2, 2] = HRoe + vRoe * cRoe

    return np.linalg.inv(R), R


def forwardTransform(meshObj, physicalValues, cell):
    return np.matmul(meshObj.connectivityData.cells[cell].invMassMatrix,
                     np.matmul(meshObj.connectivityData.cells[cell].matCell.basisMatrix.transpose(),
                               meshObj.connectivityData.cells[cell].paramSeg.weights *
                               abs(meshObj.connectivityData.cells[cell].geomCell.detJacobian) *
                               physicalValues))


def minmod(prev, current, next):

    temp_cell = np.empty((P1 + 1, nVars))

    # if var == 2:
    #     store = current[0][var]
    #     temp_cell[:, var] = 0.0
    #     temp_cell[0][var] = store
    #
    # else:

    for var in range(nVars):
        p_counter = P1
        for p in range(P1, 0, -1):
            coeff_tilde = 0
            temp_a = current[p][var]
            temp_b = (next[p - 1][var] - current[p - 1][var]) / (2 * (2 * p - 1))
            temp_c = (current[p - 1][var] - prev[p - 1][var]) / (2 * (2 * p - 1))

            # if var == 2:
            #     temp_b = (next[p - 1][var] - current[p - 1][var]) / (2 * (2 * p - 1))
            #     temp_c = (current[p - 1][var] - prev[p - 1][var]) / (2 * (2 * p - 1))
            # else:
            #     temp_b = next[p - 1][var] - current[p - 1][var]
            #     temp_c = current[p - 1][var] - prev[p - 1][var]

            if np.sign(temp_a) == np.sign(temp_b) and np.sign(temp_b) == np.sign(temp_c):
                coeff_tilde = np.sign(temp_a) * min(abs(temp_a), min(abs(temp_b), abs(temp_c)))

            if abs(temp_a - coeff_tilde) < 1.0e-6:
                break
            else:
                temp_cell[p][var] = coeff_tilde
                p_counter -= 1

        for remaining in range(p_counter, -1, -1):
            temp_cell[remaining][var] = current[remaining][var]

    return temp_cell


if __name__ == '__main__':
    """
    main()
    """

    name = "/Users/jwtan/PycharmProjects/PyDG/polyMesh/64x0_2"
    nDims = 1
    nVars = 3
    # Uniform polynomial orders in the x and y-directions for now
    P1 = 3
    P2 = 0
    # Read in the mesh
    mesh = DgMesh.constructFromPolyMeshFolder(name, nDims)
    mesh.constructShapeBasedCells(name, nVars, P1, P2)

    """
    Prototype time loop with forward Euler time-stepping
    """

    """ Set test case """
    time = 0.0
    nCells = len(mesh.connectivityData.cells)
    quadCoords = mesh.connectivityData.cells[0].GetQuadratureCoords

    """ Number of quadrature points within a single cell """
    nCoords = len(quadCoords)

    """ Numerical flux array [Number of cells in mesh, left and right numerical fluxes, number of variables]"""
    numericalFluxArray = np.empty((nCells, 2, nVars))

    """ Left and right face variable values extrapolated using basis matrices at -1 and 1 in parametric space """
    basisMatrixforF0 = Legendre1d(np.array([[-1]]), P1)
    basisMatrixforF1 = Legendre1d(np.array([[1]]), P1)
    leftFaceValueArray = np.empty((nCells, nVars))
    rightFaceValueArray = np.empty((nCells, nVars))

    test_case = "Toro1aa"
    numerical_flux = HLLCFlux

    if test_case == "Toro1a":
        rhoL = 1.0
        uL = 0.0
        pL = 1.0
        rhoR = 0.125
        uR = 0.0
        pR = 0.1
        xd = 0.5
        endTime = 0.25
        boundaryConditions = "Neumann"

    elif test_case == "Toro1aa":  # [-5, 5]
        rhoL = 1.0
        uL = 0.0
        pL = 1.0
        rhoR = 0.125
        uR = 0.0
        pR = 0.1
        xd = 0.0
        endTime = 2.0
        boundaryConditions = "Neumann"

    elif test_case == "Toro1b":
        rhoL = 1.0
        uL = 0.75
        pL = 1.0
        rhoR = 0.125
        uR = 0.0
        pR = 0.1
        xd = 0.3
        endTime = 0.2
        boundaryConditions = "Neumann"

    elif test_case == "Toro4":
        rhoL = 5.99924
        uL = 19.5975
        pL = 460.894
        rhoR = 5.99242
        uR = -6.19633
        pR = 46.0950
        xd = 0.4
        endTime = 0.035
        boundaryConditions = "Neumann"

    elif test_case == "linear_advection":
        rhoL = 1.0
        uL = 1.0
        pL = 1.0
        rhoR = 0.125
        uR = 1.0
        pR = 1.0
        xd = 0.0
        endTime = 0.1
        boundaryConditions = "Periodic"

    else:
        raise NotImplementedError

    """ Initialise primitive and conservative initial conditions """
    primL = np.array([rhoL, uL, pL])
    primR = np.array([rhoR, uR, pR])
    consL = primitiveToConservative(primL)
    consR = primitiveToConservative(primR)

    """ Set initial values using the first coefficients for constant state """
    for i in range(nCells):
        mesh.connectivityData.cells[i].solnCell.uCoeffs[0] = np.array(
            [consL if coords <= xd else consR for coords in mesh.connectivityData.cells[i].calculations.cellCentre])
        # Set physical values, no need * abs(mesh.connectivityData.cells[i].geomCell.detJacobian)
        mesh.connectivityData.cells[i].solnCell.uPhysical = np.matmul(mesh.connectivityData.cells[i].matCell.basisMatrix
                                                                      , mesh.connectivityData.cells[i].solnCell.uCoeffs)

    CFL = 0.9
    # CFL = 1 / (2 * P1 + 1)
    deltaT = 0.0
    # Constant mesh size for now
    deltax = mesh.connectivityData.cells[0].calculations.V

    """ Initialise divergence of flux and new solution coefficients """
    uCoeffsNew = np.zeros((nCells, P1 + 1, nVars))
    """ For TVD Runge Kutta, first and second stage """
    uCoeffs1 = np.zeros_like(uCoeffsNew)
    uCoeffs2 = np.zeros_like(uCoeffsNew)
    iteration = 0

    """ Start time-loop, forward Euler """
    # while iteration < 1:
    # while endTime - time > 1e-10:
    #
    #     """ Populate the faces values """
    #     for i in range(nCells):
    #         leftFaceValueArray[i] = np.matmul(basisMatrixforF0, mesh.connectivityData.cells[i].solnCell.uCoeffs)[0]
    #         rightFaceValueArray[i] = np.matmul(basisMatrixforF1, mesh.connectivityData.cells[i].solnCell.uCoeffs)[0]
    #
    #     """ Calculate time-step size """
    #     deltaT = computeDt(leftFaceValueArray, rightFaceValueArray, deltax, nCells, CFL)
    #
    #     # Increment time
    #     if time + deltaT > endTime:
    #         deltaT = endTime - time
    #     time += deltaT
    #     print("Current time: " + str(time))
    #
    #     """ Set Neumann boundary conditions, i.e., copy end cell values """
    #     numericalFluxArray[0][0] = numerical_flux(leftFaceValueArray[0], leftFaceValueArray[0]) * \
    #                                mesh.connectivityData.cells[0].calculations.faceNormals[0]
    #     numericalFluxArray[0][1] = numerical_flux(rightFaceValueArray[0], leftFaceValueArray[1]) * \
    #                                mesh.connectivityData.cells[0].calculations.faceNormals[1]
    #     numericalFluxArray[-1][0] = numerical_flux(rightFaceValueArray[-2], leftFaceValueArray[-1]) * \
    #                                 mesh.connectivityData.cells[-1].calculations.faceNormals[0]
    #     numericalFluxArray[-1][1] = numerical_flux(rightFaceValueArray[-1], rightFaceValueArray[-1]) * \
    #                                 mesh.connectivityData.cells[-1].calculations.faceNormals[1]
    #
    #     """ Calculate numerical fluxes for internal faces """
    #     for i in range(nCells - 2):
    #         numericalFluxArray[i + 1][0] = numerical_flux(rightFaceValueArray[i], leftFaceValueArray[i + 1]) \
    #                                        * mesh.connectivityData.cells[i + 1].calculations.faceNormals[0]
    #         numericalFluxArray[i + 1][1] = numerical_flux(rightFaceValueArray[i + 1], leftFaceValueArray[i + 2]) \
    #                                        * mesh.connectivityData.cells[i + 1].calculations.faceNormals[1]
    #
    #     """ Go through all cells """
    #     for i in range(nCells):
    #         """ Backward transformation from coefficient space to physical space """
    #         mesh.connectivityData.cells[i].solnCell.uPhysical = np.matmul(
    #             mesh.connectivityData.cells[i].matCell.basisMatrix, mesh.connectivityData.cells[i].solnCell.uCoeffs)
    #         """ Euler flux values at quadrature points in the cell """
    #         eulerFluxValues = np.array([eulerFlux(q) for q in mesh.connectivityData.cells[i].solnCell.uPhysical])
    #         """ Flux coefficients vector in coefficient space, f hat = (M)^-1 B^T W f """
    #         eulerFluxCoeffs = forwardTransform(mesh, eulerFluxValues, i)
    #
    #         """ Test discrete Galerkin projection operation """
    #         # print(np.matmul(mesh.connectivityData.cells[i].matCell.basisMatrix, eulerFluxMatrix2))
    #         # print(np.matmul(mesh.connectivityData.cells[i].GetStiffnessMatrix(), eulerFluxMatrix2))
    #         # print(np.matmul(basisMatrixforF0.transpose(), numericalFluxArray[i][0].reshape(-1, 1).transpose()))
    #         # print(np.matmul(basisMatrixforFace.transpose(), numericalFluxArray[i][1].reshape(-1, 1).transpose()))
    #         # + np.matmul(basisMatrixforF1.transpose(), numericalFluxArray[i][1].reshape(-1, 1).transpose()))
    #
    #         divFlux = np.matmul(mesh.connectivityData.cells[i].invMassMatrix,
    #                             (np.matmul(basisMatrixforF0.transpose(),
    #                                        numericalFluxArray[i][0].reshape(-1, 1).transpose()) +
    #                              np.matmul(basisMatrixforF1.transpose(),
    #                                        numericalFluxArray[i][1].reshape(-1, 1).transpose()) -
    #                              np.matmul(mesh.connectivityData.cells[i].stiffnessMatrix, eulerFluxCoeffs)))
    #         uCoeffsNew[i] = mesh.connectivityData.cells[i].solnCell.uCoeffs - deltaT * divFlux
    #
    #     iteration += 1
    #
    #     """ Limiting characteristic variables """
    #     avgValue = np.empty((nCells, nVars))
    #
    #     for i in range(nCells):
    #         avgValue[i] = uCoeffsNew[i][0]
    #
    #     charVars = np.empty((nCells, nCoords, nVars))
    #     charCoeffs = np.empty((nCells, P1 + 1, nVars))
    #     charCoeffsTemp = np.empty_like(charCoeffs)
    #
    #     L, R = np.empty((nCells, 3, 3)), np.empty((nCells, 3, 3))
    #
    #     for i in range(nCells - 1):
    #         L[i], R[i] = RoeMatrix(avgValue[i], avgValue[i + 1])
    #
    #     L[-1], R[-1] = RoeMatrix(avgValue[-1], avgValue[-1])
    #
    #     """ c-1, c, c+1, Neumann """
    #     charCoeffs[0] = np.matmul(L[0], uCoeffsNew[0].transpose()).transpose()
    #     charCoeffs[1] = np.matmul(L[0], uCoeffsNew[1].transpose()).transpose()
    #     charCoeffsTemp[0] = minmod(charCoeffs[0], charCoeffs[0], charCoeffs[1])
    #
    #     charCoeffs[-2] = np.matmul(L[-1], uCoeffsNew[-2].transpose()).transpose()
    #     charCoeffs[-1] = np.matmul(L[-1], uCoeffsNew[-1].transpose()).transpose()
    #     charCoeffsTemp[-1] = minmod(charCoeffs[-2], charCoeffs[-1], charCoeffs[-1])
    #
    #     for i in range(nCells - 2):
    #         charCoeffs[i] = np.matmul(L[i + 1], uCoeffsNew[i].transpose()).transpose()
    #         charCoeffs[i + 1] = np.matmul(L[i + 1], uCoeffsNew[i + 1].transpose()).transpose()
    #         charCoeffs[i + 2] = np.matmul(L[i + 1], uCoeffsNew[i + 2].transpose()).transpose()
    #         charCoeffsTemp[i + 1] = minmod(charCoeffs[i], charCoeffs[i + 1], charCoeffs[i + 2])
    #
    #     """ Back to conserved coefficients """
    #     for i in range(nCells):
    #         mesh.connectivityData.cells[i].solnCell.uCoeffs = np.matmul(R[i], charCoeffsTemp[i].transpose()).transpose()
    #
    #     """ Unlimited solution """
    #     # for i in range(nCells):
    #     #     mesh.connectivityData.cells[i].solnCell.uCoeffs = uCoeffsNew[i]

    """ Start time-loop, TVD Runge Kutta """
    while endTime - time > 1e-10:

        """ Populate the faces values """
        for i in range(nCells):
            leftFaceValueArray[i] = np.matmul(basisMatrixforF0, mesh.connectivityData.cells[i].solnCell.uCoeffs)[0]
            rightFaceValueArray[i] = np.matmul(basisMatrixforF1, mesh.connectivityData.cells[i].solnCell.uCoeffs)[0]

        """ Calculate time-step size """
        deltaT = computeDt(leftFaceValueArray, rightFaceValueArray, deltax, nCells, CFL)

        # Increment time
        if time + deltaT > endTime:
            deltaT = endTime - time
        time += deltaT
        print("Current time: " + str(time))

        """ Set Neumann boundary conditions, i.e., copy end cell values """
        numericalFluxArray[0][0] = numerical_flux(leftFaceValueArray[0], leftFaceValueArray[0]) * \
                                   mesh.connectivityData.cells[0].calculations.faceNormals[0]
        numericalFluxArray[0][1] = numerical_flux(rightFaceValueArray[0], leftFaceValueArray[1]) * \
                                   mesh.connectivityData.cells[0].calculations.faceNormals[1]
        numericalFluxArray[-1][0] = numerical_flux(rightFaceValueArray[-2], leftFaceValueArray[-1]) * \
                                    mesh.connectivityData.cells[-1].calculations.faceNormals[0]
        numericalFluxArray[-1][1] = numerical_flux(rightFaceValueArray[-1], rightFaceValueArray[-1]) * \
                                    mesh.connectivityData.cells[-1].calculations.faceNormals[1]

        """ Calculate numerical fluxes for internal faces """
        for i in range(nCells - 2):
            numericalFluxArray[i + 1][0] = numerical_flux(rightFaceValueArray[i], leftFaceValueArray[i + 1]) \
                                           * mesh.connectivityData.cells[i + 1].calculations.faceNormals[0]
            numericalFluxArray[i + 1][1] = numerical_flux(rightFaceValueArray[i + 1], leftFaceValueArray[i + 2]) \
                                           * mesh.connectivityData.cells[i + 1].calculations.faceNormals[1]

        """ Go through all cells """
        for i in range(nCells):
            """ Backward transformation from coefficient space to physical space """
            mesh.connectivityData.cells[i].solnCell.uPhysical = np.matmul(
                mesh.connectivityData.cells[i].matCell.basisMatrix, mesh.connectivityData.cells[i].solnCell.uCoeffs)
            """ Euler flux values at quadrature points in the cell """
            eulerFluxValues = np.array([eulerFlux(q) for q in mesh.connectivityData.cells[i].solnCell.uPhysical])
            """ Flux coefficients vector in coefficient space, f hat = (M)^-1 B^T W f """
            eulerFluxCoeffs = forwardTransform(mesh, eulerFluxValues, i)
            # print(np.matmul(mesh.connectivityData.cells[i].matCell.stiffnessMatrix, eulerFluxCoeffs))
            divFlux = - np.matmul(mesh.connectivityData.cells[i].invMassMatrix,
                                  (np.matmul(basisMatrixforF0.transpose(),
                                             numericalFluxArray[i][0].reshape(-1, 1).transpose()) +
                                   np.matmul(basisMatrixforF1.transpose(),
                                             numericalFluxArray[i][1].reshape(-1, 1).transpose()) -
                                   np.matmul(mesh.connectivityData.cells[i].stiffnessMatrix, eulerFluxCoeffs)))
            uCoeffs1[i] = mesh.connectivityData.cells[i].solnCell.uCoeffs + deltaT * divFlux

        """ Limiting characteristic variables """
        avgValue = np.empty((nCells, nVars))

        for i in range(nCells):
            avgValue[i] = uCoeffs1[i][0]

        charVars = np.empty((nCells, nCoords, nVars))
        charCoeffs = np.empty((nCells, P1 + 1, nVars))
        charCoeffsTemp = np.empty_like(charCoeffs)

        L, R = np.empty((nCells, 3, 3)), np.empty((nCells, 3, 3))

        for i in range(nCells - 1):
            L[i], R[i] = RoeMatrix(avgValue[i], avgValue[i + 1])

        L[-1], R[-1] = RoeMatrix(avgValue[-1], avgValue[-1])

        """ c-1, c, c+1, Neumann """
        charCoeffs[0] = np.matmul(L[0], uCoeffs1[0].transpose()).transpose()
        charCoeffs[1] = np.matmul(L[0], uCoeffs1[1].transpose()).transpose()
        charCoeffsTemp[0] = minmod(charCoeffs[0], charCoeffs[0], charCoeffs[1])

        charCoeffs[-2] = np.matmul(L[-1], uCoeffs1[-2].transpose()).transpose()
        charCoeffs[-1] = np.matmul(L[-1], uCoeffs1[-1].transpose()).transpose()
        charCoeffsTemp[-1] = minmod(charCoeffs[-2], charCoeffs[-1], charCoeffs[-1])

        for i in range(nCells - 2):
            charCoeffs[i] = np.matmul(L[i + 1], uCoeffs1[i].transpose()).transpose()
            charCoeffs[i + 1] = np.matmul(L[i + 1], uCoeffs1[i + 1].transpose()).transpose()
            charCoeffs[i + 2] = np.matmul(L[i + 1], uCoeffs1[i + 2].transpose()).transpose()
            charCoeffsTemp[i + 1] = minmod(charCoeffs[i], charCoeffs[i + 1], charCoeffs[i + 2])

        """ Back to conserved coefficients """
        for i in range(nCells):
            uCoeffs1[i] = np.matmul(R[i], charCoeffsTemp[i].transpose()).transpose()

        """ Second stage """
        for i in range(nCells):
            leftFaceValueArray[i] = np.matmul(basisMatrixforF0, uCoeffs1[i])[0]
            rightFaceValueArray[i] = np.matmul(basisMatrixforF1, uCoeffs1[i])[0]

        """ Set Neumann boundary conditions """
        numericalFluxArray[0][0] = numerical_flux(leftFaceValueArray[0], leftFaceValueArray[0]) * \
                                   mesh.connectivityData.cells[0].calculations.faceNormals[0]
        numericalFluxArray[0][1] = numerical_flux(rightFaceValueArray[0], leftFaceValueArray[1]) * \
                                   mesh.connectivityData.cells[0].calculations.faceNormals[1]
        numericalFluxArray[-1][0] = numerical_flux(rightFaceValueArray[-2], leftFaceValueArray[-1]) * \
                                    mesh.connectivityData.cells[-1].calculations.faceNormals[0]
        numericalFluxArray[-1][1] = numerical_flux(rightFaceValueArray[-1], rightFaceValueArray[-1]) * \
                                    mesh.connectivityData.cells[-1].calculations.faceNormals[1]

        """ Calculate numerical fluxes for internal faces """
        for i in range(nCells - 2):
            numericalFluxArray[i + 1][0] = numerical_flux(rightFaceValueArray[i], leftFaceValueArray[i + 1]) \
                                           * mesh.connectivityData.cells[i + 1].calculations.faceNormals[0]
            numericalFluxArray[i + 1][1] = numerical_flux(rightFaceValueArray[i + 1], leftFaceValueArray[i + 2]) \
                                           * mesh.connectivityData.cells[i + 1].calculations.faceNormals[1]

        """ Go through all cells """
        for i in range(nCells):
            """ Backward transformation from coefficient space to physical space """
            mesh.connectivityData.cells[i].solnCell.uPhysical = np.matmul(
                mesh.connectivityData.cells[i].matCell.basisMatrix, uCoeffs1[i])
            """ Euler flux values at quadrature points in the cell """
            eulerFluxValues = np.array([eulerFlux(q) for q in mesh.connectivityData.cells[i].solnCell.uPhysical])
            """ Flux coefficients vector in coefficient space, f hat = (M)^-1 B^T W f """
            eulerFluxCoeffs = forwardTransform(mesh, eulerFluxValues, i)

            divFlux = - np.matmul(mesh.connectivityData.cells[i].invMassMatrix,
                                  (np.matmul(basisMatrixforF0.transpose(),
                                             numericalFluxArray[i][0].reshape(-1, 1).transpose()) +
                                   np.matmul(basisMatrixforF1.transpose(),
                                             numericalFluxArray[i][1].reshape(-1, 1).transpose()) -
                                   np.matmul(mesh.connectivityData.cells[i].stiffnessMatrix, eulerFluxCoeffs)))
            uCoeffs2[i] = 0.75 * mesh.connectivityData.cells[i].solnCell.uCoeffs + 0.25 * uCoeffs1[i] + \
                          0.25 * deltaT * divFlux

        """ Limiting characteristic variables """
        avgValue = np.empty((nCells, nVars))

        for i in range(nCells):
            avgValue[i] = uCoeffs2[i][0]

        charVars = np.empty((nCells, nCoords, nVars))
        charCoeffs = np.empty((nCells, P1 + 1, nVars))
        charCoeffsTemp = np.empty_like(charCoeffs)

        L, R = np.empty((nCells, 3, 3)), np.empty((nCells, 3, 3))

        for i in range(nCells - 1):
            L[i], R[i] = RoeMatrix(avgValue[i], avgValue[i + 1])

        L[-1], R[-1] = RoeMatrix(avgValue[-1], avgValue[-1])

        """ c-1, c, c+1, Neumann """
        charCoeffs[0] = np.matmul(L[0], uCoeffs2[0].transpose()).transpose()
        charCoeffs[1] = np.matmul(L[0], uCoeffs2[1].transpose()).transpose()
        charCoeffsTemp[0] = minmod(charCoeffs[0], charCoeffs[0], charCoeffs[1])

        charCoeffs[-2] = np.matmul(L[-1], uCoeffs2[-2].transpose()).transpose()
        charCoeffs[-1] = np.matmul(L[-1], uCoeffs2[-1].transpose()).transpose()
        charCoeffsTemp[-1] = minmod(charCoeffs[-2], charCoeffs[-1], charCoeffs[-1])

        for i in range(nCells - 2):
            charCoeffs[i] = np.matmul(L[i + 1], uCoeffs2[i].transpose()).transpose()
            charCoeffs[i + 1] = np.matmul(L[i + 1], uCoeffs2[i + 1].transpose()).transpose()
            charCoeffs[i + 2] = np.matmul(L[i + 1], uCoeffs2[i + 2].transpose()).transpose()
            charCoeffsTemp[i + 1] = minmod(charCoeffs[i], charCoeffs[i + 1], charCoeffs[i + 2])

        """ Back to conserved coefficients """
        for i in range(nCells):
            uCoeffs2[i] = np.matmul(R[i], charCoeffsTemp[i].transpose()).transpose()

        """ Final stage """
        for i in range(nCells):
            leftFaceValueArray[i] = np.matmul(basisMatrixforF0, uCoeffs2[i])[0]
            rightFaceValueArray[i] = np.matmul(basisMatrixforF1, uCoeffs2[i])[0]

        """ Set Neumann boundary conditions """
        numericalFluxArray[0][0] = numerical_flux(leftFaceValueArray[0], leftFaceValueArray[0]) * \
                                   mesh.connectivityData.cells[0].calculations.faceNormals[0]
        numericalFluxArray[0][1] = numerical_flux(rightFaceValueArray[0], leftFaceValueArray[1]) * \
                                   mesh.connectivityData.cells[0].calculations.faceNormals[1]
        numericalFluxArray[-1][0] = numerical_flux(rightFaceValueArray[-2], leftFaceValueArray[-1]) * \
                                    mesh.connectivityData.cells[-1].calculations.faceNormals[0]
        numericalFluxArray[-1][1] = numerical_flux(rightFaceValueArray[-1], rightFaceValueArray[-1]) * \
                                    mesh.connectivityData.cells[-1].calculations.faceNormals[1]

        """ Calculate numerical fluxes for internal faces """
        for i in range(nCells - 2):
            numericalFluxArray[i + 1][0] = numerical_flux(rightFaceValueArray[i], leftFaceValueArray[i + 1]) \
                                           * mesh.connectivityData.cells[i + 1].calculations.faceNormals[0]
            numericalFluxArray[i + 1][1] = numerical_flux(rightFaceValueArray[i + 1], leftFaceValueArray[i + 2]) \
                                           * mesh.connectivityData.cells[i + 1].calculations.faceNormals[1]

        """ Go through all cells """
        for i in range(nCells):
            """ Backward transformation from coefficient space to physical space """
            mesh.connectivityData.cells[i].solnCell.uPhysical = np.matmul(
                mesh.connectivityData.cells[i].matCell.basisMatrix, uCoeffs2[i])
            """ Euler flux values at quadrature points in the cell """
            eulerFluxValues = np.array([eulerFlux(q) for q in mesh.connectivityData.cells[i].solnCell.uPhysical])
            """ Flux coefficients vector in coefficient space, f hat = (M)^-1 B^T W f """
            eulerFluxCoeffs = forwardTransform(mesh, eulerFluxValues, i)

            divFlux = - np.matmul(mesh.connectivityData.cells[i].invMassMatrix,
                                  (np.matmul(basisMatrixforF0.transpose(),
                                             numericalFluxArray[i][0].reshape(-1, 1).transpose()) +
                                   np.matmul(basisMatrixforF1.transpose(),
                                             numericalFluxArray[i][1].reshape(-1, 1).transpose()) -
                                   np.matmul(mesh.connectivityData.cells[i].stiffnessMatrix, eulerFluxCoeffs)))
            uCoeffsNew[i] = (1 / 3) * mesh.connectivityData.cells[i].solnCell.uCoeffs + (2 / 3) * uCoeffs2[i] + \
                            (2 / 3) * deltaT * divFlux

        """ Limiting characteristic variables """
        avgValue = np.empty((nCells, nVars))

        for i in range(nCells):
            avgValue[i] = uCoeffsNew[i][0]

        charVars = np.empty((nCells, nCoords, nVars))
        charCoeffs = np.empty((nCells, P1 + 1, nVars))
        charCoeffsTemp = np.empty_like(charCoeffs)

        L, R = np.empty((nCells, 3, 3)), np.empty((nCells, 3, 3))

        for i in range(nCells - 1):
            L[i], R[i] = RoeMatrix(avgValue[i], avgValue[i + 1])

        L[-1], R[-1] = RoeMatrix(avgValue[-1], avgValue[-1])

        """ c-1, c, c+1, Neumann """
        charCoeffs[0] = np.matmul(L[0], uCoeffsNew[0].transpose()).transpose()
        charCoeffs[1] = np.matmul(L[0], uCoeffsNew[1].transpose()).transpose()
        charCoeffsTemp[0] = minmod(charCoeffs[0], charCoeffs[0], charCoeffs[1])

        charCoeffs[-2] = np.matmul(L[-1], uCoeffsNew[-2].transpose()).transpose()
        charCoeffs[-1] = np.matmul(L[-1], uCoeffsNew[-1].transpose()).transpose()
        charCoeffsTemp[-1] = minmod(charCoeffs[-2], charCoeffs[-1], charCoeffs[-1])

        for i in range(nCells - 2):
            charCoeffs[i] = np.matmul(L[i + 1], uCoeffsNew[i].transpose()).transpose()
            charCoeffs[i + 1] = np.matmul(L[i + 1], uCoeffsNew[i + 1].transpose()).transpose()
            charCoeffs[i + 2] = np.matmul(L[i + 1], uCoeffsNew[i + 2].transpose()).transpose()
            charCoeffsTemp[i + 1] = minmod(charCoeffs[i], charCoeffs[i + 1], charCoeffs[i + 2])

        """ Back to conserved coefficients """
        for i in range(nCells):
            mesh.connectivityData.cells[i].solnCell.uCoeffs = np.matmul(R[i], charCoeffsTemp[i].transpose()).transpose()

        # for i in range(nCells):
        #     mesh.connectivityData.cells[i].solnCell.uCoeffs = uCoeffsNew[i]

    """ Plotting """
    pltPhysical = np.zeros((nCells, nCoords, nVars))
    for i in range(nCells):
        pltPhysical[i] = np.matmul(mesh.connectivityData.cells[i].matCell.basisMatrix,
                                   mesh.connectivityData.cells[i].solnCell.uCoeffs)

    pltCoords = np.array([mesh.connectivityData.cells[i].GetQuadratureCoords for i in range(nCells)]).reshape(-1)
    plt.plot(pltCoords, (pltPhysical[:, :, 0]).reshape(-1))
    plt.plot(pltCoords, (pltPhysical[:, :, 1] / pltPhysical[:, :, 0]).reshape(-1))
    plt.plot(pltCoords, (0.4 * (pltPhysical[:, :, 2] - 0.5 * pltPhysical[:, :, 0] * pltPhysical[:, :, 1] *
                                pltPhysical[:, :, 1])).reshape(-1))
    plt.grid()
    plt_name = test_case + "_" + numerical_flux.__name__ + "_" + "P" + str(P1) + "_" + str(endTime) + "_" + "limited"
    # plt.savefig(plt_name + '.png', dpi=100)
    # np.savetxt(plt_name + "_coords.dat", pltCoords, delimiter=',')
    # np.savetxt(plt_name + "_values.dat", pltPhysical[:, :, 0].reshape(-1), delimiter=',')
    plt.show()

    # for face in mesh.connectivityData.cells[i].geomData.neighbourLabels:
    #
    #     if face is not None:
