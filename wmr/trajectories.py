"""Reference trajectory generators (paper Section V).

Each trajectory returns the reference pose p_r = [x_r, y_r, theta_r] and the
reference body velocities v_r (linear) and w_r (angular) as functions of time.
"""
import numpy as np


class Trajectory:
    """Base interface: a trajectory maps time -> (pose, v_r, w_r)."""

    def state(self, t):
        raise NotImplementedError

    def __call__(self, t):
        return self.state(t)


class StraightLine(Trajectory):
    """Straight-line reference (paper Section V).

        x = x0 + v_r t cos(alpha)
        y = y0 + v_r t sin(alpha)

    Constant heading theta_r = alpha, v_r constant, w_r = 0.
    """

    def __init__(self, x0=1.8, y0=0.2, alpha=np.pi / 6, vr=2.0):
        self.x0 = x0
        self.y0 = y0
        self.alpha = alpha
        self.vr = vr

    def state(self, t):
        x = self.x0 + self.vr * t * np.cos(self.alpha)
        y = self.y0 + self.vr * t * np.sin(self.alpha)
        theta = self.alpha
        return np.array([x, y, theta]), self.vr, 0.0


class Circle(Trajectory):
    """Circular reference (paper Section V).

        x = R cos(w_r t)
        y = R sin(w_r t)

    With R = 2 m and w_r = 1 rad/s the linear speed is v_r = R*w_r = 2 m/s.
    Heading is tangent to the circle: theta_r = w_r t + pi/2.
    """

    def __init__(self, R=2.0, wr=1.0):
        self.R = R
        self.wr = wr
        self.vr = R * wr

    def state(self, t):
        x = self.R * np.cos(self.wr * t)
        y = self.R * np.sin(self.wr * t)
        theta = self.wr * t + np.pi / 2.0
        return np.array([x, y, theta]), self.vr, self.wr


def make_trajectory(name):
    if name == "line":
        return StraightLine()
    if name == "circle":
        return Circle()
    raise ValueError(f"unknown trajectory: {name!r} (use 'line' or 'circle')")
