"""
Solver options and case setup

Author: JWT
"""
import numpy as np

"""======SPACE-TIME DOMAIN======"""
NO_OF_DIMENSIONS = 1
DOMAIN_BOUNDARIES = [[-1.], [1.]]
NO_OF_NODES = [101]
ELEM_SHAPE = "S"
BOUNDARY_CONDITIONS_TYPE = ["Periodic", "Periodic"]
BOUNDARY_CONDITIONS_VALUES = [1., -1.]
FINAL_TIME = 0.5

"""=====USER-DEFINED DOMAIN====="""
USER_INPUT = False
INPUT_NAME = ""

"""=======SOLVER OPTIONS========"""
CFL = 0.9
EQUATION_TYPE = "ADR"
VEL_FIELD = [1.]
KAPPA = 0.05
NO_OF_VARIABLES = 1
P1 = 1
