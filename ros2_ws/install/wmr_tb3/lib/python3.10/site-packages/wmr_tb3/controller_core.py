"""Self-contained kinematic backstepping + adaptive slip controller.

Vendored from the validated `wmr` reproduction (Lu et al., ICIEA 2024) so the
ROS 2 package has no external dependency.  Defaults are set for TurtleBot3
Burger (r = 0.033 m, wheel separation 0.16 m -> b = 0.08 m, centroid offset
d ~ 0 since the two drive wheels carry the body, with a passive caster).
"""
import math


class TB3Params:
    # geometry
    r = 0.033       # wheel radius [m]
    b = 0.08        # half wheel separation [m]
    d = 0.0         # centroid offset from the drive-wheel axle [m]
    # kinematic controller / adaptive gains (eq.12-14)
    Hx = 1.5
    Hy = 1.5
    Hs = 0.2
    lam = 0.5
    rho1 = 10.0
    rho2 = 4.0
    # actuation limits (TurtleBot3 Burger)
    v_max = 0.22    # [m/s]
    w_max = 2.84    # [rad/s]


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def pose_error(pose, ref):
    """Tracking error in the robot body frame (reference - actual).

    pose, ref = (x, y, theta).
    """
    th = pose[2]
    dx = ref[0] - pose[0]
    dy = ref[1] - pose[1]
    xe = math.cos(th) * dx + math.sin(th) * dy
    ye = -math.sin(th) * dx + math.cos(th) * dy
    the = wrap(ref[2] - th)
    return (xe, ye, the)


class KinematicAdaptiveController:
    """Kinematic virtual control law (eq.12) + online slip adaptation (eq.13/14).

    estimate i_k = 1/(1 - s_k);  i = 1 means "no slip compensation".
    """

    def __init__(self, p=TB3Params, adaptive=True):
        self.p = p
        self.adaptive = adaptive
        self.iL = 1.0
        self.iR = 1.0

    def virtual_velocity(self, e, vr, wr, w_meas):
        xe, ye, the = e
        p = self.p
        vc = vr * math.cos(the) + p.Hx * (xe + p.d * (1 - math.cos(the))) \
            - p.Hs * the * w_meas
        wc = wr + vr * (
            p.Hy * (1 - p.lam) * (ye - p.d * math.sin(the) + p.Hs * the)
            + (p.lam / p.Hs) * math.sin(the))
        return vc, wc

    def adapt(self, e, vc, wc, dt):
        if not self.adaptive:
            return
        xe, ye, the = e
        p = self.p
        He2 = (1.0 / p.Hy) * math.sin(the)
        He1 = p.Hs * (ye - p.d * math.sin(the) + xe * the
                      + p.d * the * (1 - math.cos(the)) + p.Hs * the)
        base = p.b * xe + p.b * p.d * (1 - math.cos(the))
        iR_dot = (1.0 / (2 * p.b)) * p.rho2 * (vc + p.b * wc) * (base + He1 + He2)
        iL_dot = (1.0 / (2 * p.b)) * p.rho1 * (vc - p.b * wc) * (base - He1 - He2)
        # clip i = 1/(1-s) to a physical range (slip in ~[-1, 0.67])
        self.iR = min(max(self.iR + iR_dot * dt, 0.5), 3.0)
        self.iL = min(max(self.iL + iL_dot * dt, 0.5), 3.0)

    def wheel_commands(self, vc, wc):
        """Virtual velocity -> slip-compensated wheel angular speeds (eq.11)."""
        p = self.p
        wL = (self.iL / p.r) * (vc - p.b * wc)
        wR = (self.iR / p.r) * (vc + p.b * wc)
        return wL, wR

    def cmd_vel(self, vc, wc):
        """Slip-compensated body command for TurtleBot3's cmd_vel.

        We invert the wheel speeds back to (v, w) through the robot's own r, b so
        TB3's internal wheel-velocity loop reproduces the i^-inflated wheel
        speeds (the DYNAMIXEL velocity loop ignores ground slip, so the wheels
        actually spin faster and compensate the slip).
        """
        p = self.p
        wL, wR = self.wheel_commands(vc, wc)
        v = p.r * (wR + wL) / 2.0
        w = p.r * (wR - wL) / (2.0 * p.b)
        # saturate to actuator limits
        v = max(min(v, p.v_max), -p.v_max)
        w = max(min(w, p.w_max), -p.w_max)
        return v, w

    @property
    def slip_estimate(self):
        return 1.0 - 1.0 / self.iL, 1.0 - 1.0 / self.iR
