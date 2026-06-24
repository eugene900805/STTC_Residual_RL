"""Gymnasium environment for WMR slipping trajectory tracking.

The agent drives the same slipping plant the paper controller drives, so the two
can be compared head-to-head.

Observation (slip is *hidden* -- the agent must reject it from error feedback,
just as the paper's adaptive law does):
    [xe, ye, sin(theta_e), cos(theta_e), vr, wr]

Action (body velocity command, mapped to wheel speeds *without* slip
compensation; the plant then applies the true slip):
    [v_cmd, w_cmd]  in normalized [-1, 1], scaled to [+-V_MAX, +-W_MAX]

Reward: negative quadratic tracking error minus a small control penalty.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from wmr.params import RobotParams
from wmr.model import KinematicWMRPlant, body_to_wheel, wheel_to_body
from wmr.trajectories import StraightLine, Circle
from wmr.controller import pose_error

V_MAX = 4.0     # max commanded linear velocity [m/s]
W_MAX = 4.0     # max commanded angular velocity [rad/s]


class WMRTrackingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, dt=0.05, horizon=20.0, randomize=True,
                 traj="mix", slip_profile=None, pose0=None, seed=None):
        super().__init__()
        self.p = RobotParams()
        self.dt = dt
        self.horizon = horizon
        self.max_steps = int(round(horizon / dt))
        self.randomize = randomize
        self.traj_mode = traj                 # 'line' | 'circle' | 'mix'
        self.fixed_slip = slip_profile        # callable(t)->(sL,sR) or None
        self.fixed_pose0 = pose0
        self.rng = np.random.default_rng(seed)

        # obs: [xe, ye, sin th_e, cos th_e, vr, wr]
        high = np.array([np.inf, np.inf, 1.0, 1.0, np.inf, np.inf], np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    # ------------------------------------------------------------------ #
    def _make_trajectory(self):
        if self.traj_mode == "line":
            return StraightLine()
        if self.traj_mode == "circle":
            return Circle()
        return StraightLine() if self.rng.random() < 0.5 else Circle()

    def _sample_slip(self):
        """Random piecewise-constant slip with one step change (training)."""
        sL0, sR0 = self.rng.uniform(0.0, 0.25, size=2)
        sL1, sR1 = self.rng.uniform(0.0, 0.30, size=2)
        tL = self.rng.uniform(0.3, 0.7) * self.horizon
        tR = self.rng.uniform(0.3, 0.7) * self.horizon

        def slip(t):
            sR = sR0 if t < tR else sR1
            sL = sL0 if t < tL else sL1
            return sL, sR
        return slip

    # ------------------------------------------------------------------ #
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.traj = self._make_trajectory()
        self.slip = self.fixed_slip or (
            self._sample_slip() if self.randomize else (lambda t: (0.1, 0.1)))

        ref0, _, _ = self.traj.state(0.0)
        if self.fixed_pose0 is not None:
            pose0 = np.array(self.fixed_pose0, float)
        elif self.randomize:
            off = self.rng.uniform(-0.4, 0.4, size=3)
            off[2] = self.rng.uniform(-0.6, 0.6)
            pose0 = ref0 + off
        else:
            pose0 = ref0 + np.array([-0.2, -0.2, 0.3])

        self.plant = KinematicWMRPlant(self.p, pose0=pose0)
        self.t = 0.0
        self.step_i = 0
        return self._obs(), {}

    def _obs(self):
        ref, vr, wr = self.traj.state(self.t)
        e = pose_error(self.plant.pose, ref)
        self._e = e
        return np.array([e[0], e[1], np.sin(e[2]), np.cos(e[2]), vr, wr],
                        dtype=np.float32)

    # ------------------------------------------------------------------ #
    def step(self, action):
        a = np.clip(action, -1.0, 1.0)
        v_cmd = float(a[0]) * V_MAX
        w_cmd = float(a[1]) * W_MAX

        # body command -> wheel speeds (no slip compensation), plant adds slip
        wL, wR = body_to_wheel(v_cmd, w_cmd, self.p)
        sL, sR = self.slip(self.t)
        self.plant.step(wL, wR, sL, sR, self.dt)
        self.t += self.dt
        self.step_i += 1

        obs = self._obs()
        xe, ye, the = self._e
        pos_err2 = xe * xe + ye * ye
        reward = -(pos_err2 + 0.3 * the * the) - 1e-3 * (a[0] ** 2 + a[1] ** 2)

        terminated = bool(np.sqrt(pos_err2) > 3.0)   # diverged
        if terminated:
            reward -= 10.0
        truncated = self.step_i >= self.max_steps
        return obs, float(reward), terminated, truncated, {}
