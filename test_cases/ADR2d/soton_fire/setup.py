"""
Solver options and case setup

Author: JWT
"""


"""======SPACE-TIME DOMAIN======"""
NO_OF_DIMENSIONS = 2
DOMAIN_BOUNDARIES = [[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]]
NO_OF_NODES = [65, 65]
ELEM_SHAPE = "T"
BOUNDARY_CONDITIONS_TYPE = ["Periodic", "Periodic", "Periodic", "Periodic"]
BOUNDARY_CONDITIONS_VALUES = [1., 1., -1., -1.]
FINAL_TIME = 10. * 3600.

"""=====USER-DEFINED DOMAIN====="""
USER_INPUT = True
INPUT_NAME = '5'

"""=======SOLVER OPTIONS========"""
CFL = 0.5
EQUATION_TYPE = "ADR"
VEL_FIELD = [0., -10.]
KAPPA = 10000.
NO_OF_VARIABLES = 1
P1 = 1
P2 = 1
SOTON_FIRE = 8. * 3600.
