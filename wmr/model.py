"""WMR plant models with longitudinal wheel slip.

Two fidelity levels are provided:

* ``KinematicWMRPlant`` -- the robot is commanded by wheel angular velocities
  (w_L, w_R); longitudinal slip degrades the effective ground velocity of each
  wheel (eq.1).  This is the cleanest plant for demonstrating the paper's core
  contribution: the adaptive slip estimator.

* ``DynamicWMRPlant`` -- adds the wheel/body velocity dynamics so the robot is
  commanded by wheel torques (tau_L, tau_R), exercising the dynamic backstepping
  controller (eq.18).  A reduced, structurally-faithful form of the paper's
  Lagrange model (eq.7/8) is used.

Both keep the pose at the centroid G, which sits a distance ``d`` ahead of the
drive-wheel axle along the heading -- consistent with the controller's use of d.
"""
import numpy as np

from .params import RobotParams


class SlipProfile:
    """Time-varying true longitudinal slip ratios s_L(t), s_R(t).

    The paper changes the left/right slip parameters at t = 30 s and t = 50 s to
    demonstrate robustness.  We reproduce this with piecewise-constant steps.
    """

    def __init__(self, sL0=0.10, sR0=0.10, sL1=0.25, sR1=0.20,
                 t_left=50.0, t_right=30.0):
        self.sL0, self.sR0 = sL0, sR0
        self.sL1, self.sR1 = sL1, sR1
        self.t_left, self.t_right = t_left, t_right

    def __call__(self, t):
        sR = self.sR0 if t < self.t_right else self.sR1
        sL = self.sL0 if t < self.t_left else self.sL1
        return sL, sR


def wheel_to_body(wL, wR, sL, sR, p: RobotParams):
    """Map (possibly slipping) wheel angular velocities to body (v, w).

    Effective ground velocity of each wheel (eq.1):  v_k = r * w_k * (1 - s_k).
    Differential drive:  v = (v_R + v_L)/2,  w = (v_R - v_L)/(2 b).
    """
    vL = p.r * wL * (1.0 - sL)
    vR = p.r * wR * (1.0 - sR)
    v = 0.5 * (vR + vL)
    w = (vR - vL) / (2.0 * p.b)
    return v, w


def body_to_wheel(v, w, p: RobotParams):
    """Inverse map (no slip): body (v, w) -> wheel angular velocities."""
    vL = v - p.b * w
    vR = v + p.b * w
    return vL / p.r, vR / p.r


def centroid_kinematics(theta, v, w, d):
    """Velocity of the centroid G (offset d ahead of the axle along heading)."""
    xdot = v * np.cos(theta) - d * w * np.sin(theta)
    ydot = v * np.sin(theta) + d * w * np.cos(theta)
    return np.array([xdot, ydot, w])


class KinematicWMRPlant:
    """Kinematic plant: input = wheel angular velocities, slip applied externally."""

    def __init__(self, params: RobotParams = None, pose0=(0.0, 0.0, 0.0)):
        self.p = params or RobotParams()
        self.pose = np.array(pose0, dtype=float)   # [xG, yG, theta]
        self.v = 0.0
        self.w = 0.0

    def step(self, wL, wR, sL, sR, dt):
        """Advance one step given wheel commands and the *true* slip ratios."""
        v, w = wheel_to_body(wL, wR, sL, sR, self.p)
        self.v, self.w = v, w
        # RK4 on the centroid kinematics (theta couples x,y)
        k1 = centroid_kinematics(self.pose[2], v, w, self.p.d)
        k2 = centroid_kinematics(self.pose[2] + 0.5 * dt * k1[2], v, w, self.p.d)
        k3 = centroid_kinematics(self.pose[2] + 0.5 * dt * k2[2], v, w, self.p.d)
        k4 = centroid_kinematics(self.pose[2] + dt * k3[2], v, w, self.p.d)
        self.pose = self.pose + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return self.pose.copy()


class DynamicWMRPlant:
    """Dynamic plant: input = wheel torques (tau_L, tau_R).

    Wheel rotation dynamics (paper eq.7, last two rows):
        I_w * wdot_k = tau_k - r * F_xk

    The longitudinal tire force F_xk is the reaction that the ground exerts on
    the wheel.  We model it as a viscous traction coupling the wheel speed to the
    ground-contact speed it should produce given the current slip ratio (eq.1):

        v_k^target = r * w_k * (1 - s_k)        (ground speed the wheel rolls at)
        F_xk = C_t * (r * w_k - v_k^target)/r = C_t * w_k * s_k

    The body then accelerates under the two traction forces.  The wheel-speed ->
    body-velocity map (eq.1, differential drive) closes the loop, so torque
    propagates to motion through the wheel inertia exactly as in the cascade of
    the paper (kinematic layer commands wheel speeds, dynamic layer realises
    them with torque).
    """

    def __init__(self, params: RobotParams = None, pose0=(0.0, 0.0, 0.0),
                 Ct=120.0, n_sub=40):
        self.p = params or RobotParams()
        self.pose = np.array(pose0, dtype=float)
        self.wL = 0.0   # wheel angular velocities (states)
        self.wR = 0.0
        self.v = 0.0
        self.w = 0.0
        self.Ct = Ct          # traction stiffness [N s]
        self.n_sub = n_sub    # substeps per control step (the contact is stiff)
        self._m = self.p.m + 2.0 * self.p.m_w + 2.0 * self.p.I_w / self.p.r ** 2
        self._I = self.p.I + 2.0 * self.p.m_w * self.p.b ** 2 \
            + 2.0 * self.p.I_w * self.p.b ** 2 / self.p.r ** 2

    def micro_step(self, tauL, tauR, sL, sR, h):
        """Integrate the physical dynamics for one fine step of length h."""
        p = self.p
        # --- wheel rotational dynamics: I_w wdot = tau - r F_x ---
        # traction force pulls wheel speed toward the body-consistent speed
        FxL = self.Ct * (p.r * self.wL * (1.0 - sL) - (self.v - p.b * self.w)) / p.r
        FxR = self.Ct * (p.r * self.wR * (1.0 - sR) - (self.v + p.b * self.w)) / p.r
        self.wL += (tauL - p.r * FxL) / p.I_w * h
        self.wR += (tauR - p.r * FxR) / p.I_w * h
        # --- body dynamics driven by the same traction forces ---
        self.v += (FxL + FxR) / self._m * h
        self.w += (p.b * (FxR - FxL)) / self._I * h
        # --- integrate centroid pose ---
        self.pose = self.pose + h * centroid_kinematics(
            self.pose[2], self.v, self.w, p.d)
        return self.pose.copy()

    def step(self, tauL, tauR, sL, sR, dt):
        """Apply a constant torque over dt (n_sub fine substeps)."""
        h = dt / self.n_sub
        for _ in range(self.n_sub):
            self.micro_step(tauL, tauR, sL, sR, h)
        return self.pose.copy()
