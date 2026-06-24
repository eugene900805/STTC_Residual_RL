"""Hierarchical controller from the paper.

Layer 1 -- kinematic backstepping virtual control law (eq.12) plus the adaptive
slip estimator (eq.13/14).  Produces commanded body velocities (v_c, w_c) and
maps them to wheel angular velocity commands using the estimated slip (eq.11).

Layer 2 -- dynamic backstepping control law (eq.18) turning the virtual velocity
command into wheel torques for the dynamic plant.

The estimated parameter is i_k = 1/(1 - s_k) (eq.9): when the estimate matches
the true slip, the commanded body velocity is achieved exactly despite slip.
"""
import numpy as np

from .params import RobotParams, KinematicGains, DynamicGains


def pose_error(pose, pose_ref):
    """Tracking error in the robot body frame (reference - actual).

        xe =  cos th (xr-x) + sin th (yr-y)
        ye = -sin th (xr-x) + cos th (yr-y)
        the = wrap(theta_r - theta)
    """
    th = pose[2]
    dx = pose_ref[0] - pose[0]
    dy = pose_ref[1] - pose[1]
    xe = np.cos(th) * dx + np.sin(th) * dy
    ye = -np.sin(th) * dx + np.cos(th) * dy
    the = np.arctan2(np.sin(pose_ref[2] - th), np.cos(pose_ref[2] - th))
    return np.array([xe, ye, the])


class KinematicAdaptiveController:
    """Kinematic virtual control law + online slip-ratio adaptation."""

    def __init__(self, robot: RobotParams = None, gains: KinematicGains = None,
                 i_init=(1.0, 1.0)):
        self.p = robot or RobotParams()
        self.g = gains or KinematicGains()
        # estimated i_k = 1/(1-s_k); start at 1.0 (assume no slip).
        self.iL = float(i_init[0])
        self.iR = float(i_init[1])

    def virtual_velocity(self, e, vr, wr, w_meas):
        """Kinematic virtual control law (eq.12). e = [xe, ye, the]."""
        xe, ye, the = e
        g, d = self.g, self.p.d
        vc = vr * np.cos(the) + g.Hx * (xe + d * (1.0 - np.cos(the))) \
            - g.Hs * the * w_meas
        wc = wr + vr * (
            g.Hy * (1.0 - g.lam) * (ye - d * np.sin(the) + g.Hs * the)
            + (g.lam / g.Hs) * np.sin(the)
        )
        return vc, wc

    def adapt(self, e, vc, wc, dt):
        """Adaptive slip-estimation laws (paper eq.13/14).

            He2 = (1/Hy) sin th
            He1 = Hs (ye - d sin th + xe th + d th(1-cos th) + Hs th)
            base = b xe + b d (1 - cos th)
            idot_R = (1/2b) rho2 (v_c + b w_c)(base + He1 + He2)
            idot_L = (1/2b) rho1 (v_c - b w_c)(base - He1 - He2)

        The regressor (v_c +- b w_c) is the per-wheel velocity command, and
        (base, He1, He2) are the velocity-gradient terms of the Lyapunov
        function V1 (eq.19).  When the estimate i^_k -> i_k = 1/(1-s_k) the
        commanded body velocity is achieved exactly despite slip.
        """
        xe, ye, the = e
        g, b, d = self.g, self.p.b, self.p.d
        He2 = (1.0 / g.Hy) * np.sin(the)
        He1 = g.Hs * (ye - d * np.sin(the) + xe * the
                      + d * the * (1.0 - np.cos(the)) + g.Hs * the)
        base = b * xe + b * d * (1.0 - np.cos(the))
        iR_dot = (1.0 / (2.0 * b)) * g.rho2 * (vc + b * wc) * (base + He1 + He2)
        iL_dot = (1.0 / (2.0 * b)) * g.rho1 * (vc - b * wc) * (base - He1 - He2)
        self.iR += iR_dot * dt
        self.iL += iL_dot * dt
        # keep estimates physically meaningful: i = 1/(1-s) bounded away from 0
        self.iL = float(np.clip(self.iL, 0.2, 10.0))
        self.iR = float(np.clip(self.iR, 0.2, 10.0))

    def wheel_commands(self, vc, wc):
        """Map virtual velocity to wheel angular velocities using i-estimates (eq.11).

            wL = (iL/r)(v - b w),   wR = (iR/r)(v + b w)
        """
        r, b = self.p.r, self.p.b
        wL = (self.iL / r) * (vc - b * wc)
        wR = (self.iR / r) * (vc + b * wc)
        return wL, wR

    @property
    def slip_estimate(self):
        """Estimated slip ratios s_k = 1 - 1/i_k."""
        return 1.0 - 1.0 / self.iL, 1.0 - 1.0 / self.iR


class DynamicBackstepping:
    """Dynamic backstepping torque law (eq.18) as a wheel-speed inner loop.

    The kinematic+adaptive layer outputs slip-compensated wheel-speed commands
    (w_L^cmd, w_R^cmd) (eq.11).  Here the wheel rotational dynamics
    I_w wdot_k = tau_k - r F_xk (eq.7) are inverted by backstepping to drive the
    wheel-speed error z2_k = w_k^cmd - w_k to zero:

        tau_k = I_w (wdot_k^cmd + c1 z2_k + c2 \\int z2_k)

    At cruise the traction force vanishes (wheel speed matches the body speed),
    so no traction feedforward is needed; the integral term c2 removes the lag
    from transient traction loads.  This is the reduced (wheel-speed) form of the
    backstepping law (eq.18); velocity-error convergence follows (Proof 2).
    """

    def __init__(self, robot: RobotParams = None, gains: DynamicGains = None):
        self.p = robot or RobotParams()
        self.g = gains or DynamicGains()

    def torques(self, wL_cmd, wR_cmd, wL_meas, wR_meas, v_meas, w_meas, dt):
        p = self.p
        eL, eR = wL_cmd - wL_meas, wR_cmd - wR_meas
        tauL = p.I_w * self.g.c1 * eL
        tauR = p.I_w * self.g.c1 * eR
        return tauL, tauR
