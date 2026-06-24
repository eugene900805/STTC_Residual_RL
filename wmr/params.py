"""Physical and controller parameters for the WMR slipping-tracking simulation.

All values follow the paper (Lu et al., ICIEA 2024, "Slipping trajectory
tracking control of wheeled mobile robot based on dynamics model"),
Section V "Simulation".
"""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class RobotParams:
    """Geometric / inertial parameters of the differential-drive WMR."""
    b: float = 0.15      # half the distance between the two rear wheels [m]
    d: float = 0.2       # distance from centroid G to the drive-wheel axle [m]
    r: float = 0.125     # wheel radius [m]

    # --- inertial parameters (used by the dynamic layer) ---
    m: float = 10.0      # total mass of the WMR body [kg]
    m_w: float = 0.5     # mass of one drive wheel + motor rotor [kg]
    I: float = 0.5       # moment of inertia of body about G [kg m^2]
    I_w: float = 0.05    # moment of inertia of one wheel about its axle [kg m^2]


@dataclass
class TireParams:
    """Brush-model tire parameters (paper Section V)."""
    l_w: float = 0.001   # half contact length [m]
    b_w: float = 0.01    # contact width [m]
    beta: float = 0.0873  # side-slip angle [rad]
    Kx: float = 2.0
    Ky: float = 5.0
    Fz: float = 20.0     # vertical load [N]
    mu: float = 0.2      # friction coefficient


@dataclass
class KinematicGains:
    """Gains of the kinematic virtual control law (eq.12) and adaptive laws."""
    Hx: float = 5.0
    Hy: float = 5.0
    Hs: float = 0.2
    lam: float = 0.5     # lambda in [0, 1]
    rho1: float = 10.0   # adaptation gain, left wheel
    rho2: float = 4.0    # adaptation gain, right wheel


@dataclass
class DynamicGains:
    """Gains of the dynamic backstepping control law (eq.18).

    The paper quotes c1=39, c2=16 for the (v, w)-space law.  Our reduced inner
    loop tracks wheel speed directly, so c1 is re-tuned high enough that the
    inner-loop bandwidth sits well above the kinematic outer loop -- preserving
    the time-scale separation the hierarchical design assumes.  Too low a c1
    (e.g. 150) leaves a heading limit-cycle; c1=800 removes it and recovers the
    same tracking accuracy as the ideal-velocity kinematic layer.
    """
    c1: float = 800.0
    c2: float = 16.0
