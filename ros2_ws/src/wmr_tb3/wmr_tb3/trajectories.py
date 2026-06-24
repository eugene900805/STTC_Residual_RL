"""Reference trajectories scaled to TurtleBot3 velocity limits."""
import math


class StraightLine:
    """x = x0 + vr t cos(a), y = y0 + vr t sin(a); heading a, wr = 0."""

    def __init__(self, x0=0.0, y0=0.0, alpha=0.0, vr=0.12):
        self.x0, self.y0, self.alpha, self.vr = x0, y0, alpha, vr

    def state(self, t):
        x = self.x0 + self.vr * t * math.cos(self.alpha)
        y = self.y0 + self.vr * t * math.sin(self.alpha)
        return (x, y, self.alpha), self.vr, 0.0


class Circle:
    """x = cx + R cos(wr t), y = cy + R sin(wr t); v = R*wr."""

    def __init__(self, R=1.0, wr=0.12, cx=None, cy=0.0):
        self.R, self.wr, self.cy = R, wr, cy
        # default centre so the circle starts at the world origin region
        self.cx = cx if cx is not None else 0.0
        self.vr = R * wr

    def state(self, t):
        x = self.cx + self.R * math.cos(self.wr * t)
        y = self.cy + self.R * math.sin(self.wr * t)
        theta = self.wr * t + math.pi / 2.0
        return (x, y, theta), self.vr, self.wr


def make_trajectory(name, **kw):
    if name == 'line':
        return StraightLine(**kw)
    if name == 'circle':
        return Circle(**kw)
    raise ValueError("trajectory must be 'line' or 'circle'")
