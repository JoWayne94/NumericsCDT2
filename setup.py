"""
Solver options and case setup

Author: JWT
"""


"""======SPACE-TIME DOMAIN======"""
NO_OF_DIMENSIONS = 2
DOMAIN_BOUNDARIES = [[0., 0.], [1., 0.], [1., 1.], [0., 1.]]
NO_OF_NODES = [65, 65]
ELEM_SHAPE = "T"
BOUNDARY_CONDITIONS_TYPE = ["Dirichlet", "Neumann", "Neumann", "Neumann"]
BOUNDARY_CONDITIONS_VALUES = [0., 0., 0., 0.]
FINAL_TIME = 8. * 3600.

"""=====USER-DEFINED DOMAIN====="""
USER_INPUT = True
INPUT_NAME = "5"

"""=======SOLVER OPTIONS========"""
CFL = 0.5
EQUATION_TYPE = "ADR"
VEL_FIELD = [0., -10.]
KAPPA = 1000.
NO_OF_VARIABLES = 1
P1 = 1
P2 = 1
