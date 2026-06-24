"""Reproduction of Lu et al. (ICIEA 2024): slipping trajectory tracking control
of a wheeled mobile robot based on a dynamics model.

Modules
-------
params        physical / controller parameters
trajectories  reference trajectory generators (line, circle)
model         WMR plant models with longitudinal slip + slip profile
controller    backstepping kinematic + adaptive slip + dynamic torque control
simulate      simulation loop and plotting
"""
from . import params, trajectories, model, controller, simulate  # noqa: F401
