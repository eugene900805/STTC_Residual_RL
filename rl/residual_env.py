"""Residual-RL environment for WMR slipping trajectory tracking.

The paper's kinematic backstepping + adaptive slip controller is the *base*
controller and does most of the work.  The RL agent only outputs a small
residual correction added to the base virtual velocity:

    v_cmd = v_c^base + dv_RL
    w_cmd = w_c^base + dw_RL

The adaptive slip law keeps running underneath, so the agent learns to fix what
the base controller leaves behind (e.g. the circle steady-state residual) while
inheriting its stability.

Observation: [xe, ye, sin th_e, cos th_e, vr, wr, v_c, w_c]   (base cmd included)
Action:      [dv, dw] in [-1,1], scaled to [+-RES_V, +-RES_W]
Reward:      negative quadratic tracking error - small residual penalty
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from wmr.params import RobotParams, KinematicGains
from wmr.model import KinematicWMRPlant, SlipProfile
from wmr.trajectories import StraightLine, Circle
from wmr.controller import KinematicAdaptiveController, pose_error

RES_V = 1.0     # max residual linear velocity [m/s]
RES_W = 1.0     # max residual angular velocity [rad/s]


class WMRResidualEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, dt=0.05, horizon=20.0, randomize=True, traj="mix",
                 slip_profile=None, pose0=None, res_penalty=0.02,
                 base_adaptive=True, seed=None):
        super().__init__()
        self.p = RobotParams()
        self.gains = KinematicGains()
        # base_adaptive=False -> base = paper backstepping (eq.12) with NO slip
        # compensation (i^=1); the RL residual must learn to reject the slip,
        # i.e. it takes over the job of the adaptive law (option B).
        self.base_adaptive = base_adaptive
        self.dt = dt
        self.horizon = horizon
        self.max_steps = int(round(horizon / dt))
        self.randomize = randomize
        self.traj_mode = traj
        self.fixed_slip = slip_profile
        self.fixed_pose0 = pose0
        self.res_penalty = res_penalty
        self.rng = np.random.default_rng(seed)

        high = np.array([np.inf] * 8, np.float32)
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
        self.ctrl = KinematicAdaptiveController(self.p, self.gains)
        self.t = 0.0
        self.step_i = 0
        self._refresh()
        return self._obs, {}

    def _refresh(self):
        """Recompute pose error and the base controller virtual velocity."""
        ref, vr, wr = self.traj.state(self.t)
        e = pose_error(self.plant.pose, ref)
        vc, wc = self.ctrl.virtual_velocity(e, vr, wr, self.plant.w)
        self.e, self.vr, self.wr, self.vc, self.wc = e, vr, wr, vc, wc
        self._obs = np.array([e[0], e[1], np.sin(e[2]), np.cos(e[2]),
                              vr, wr, vc, wc], dtype=np.float32)

    # ------------------------------------------------------------------ #
    def step(self, action):
        a = np.clip(action, -1.0, 1.0)
        dv = float(a[0]) * RES_V
        dw = float(a[1]) * RES_W

        # advance the adaptive slip estimate first (as in the paper's loop),
        # then map the residual-augmented command to wheels with the updated i^
        if self.base_adaptive:
            self.ctrl.adapt(self.e, self.vc, self.wc, self.dt)
        v_cmd = self.vc + dv
        w_cmd = self.wc + dw
        wL, wR = self.ctrl.wheel_commands(v_cmd, w_cmd)

        sL, sR = self.slip(self.t)
        self.plant.step(wL, wR, sL, sR, self.dt)
        self.t += self.dt
        self.step_i += 1
        self._refresh()

        xe, ye, the = self.e
        pos_err2 = xe * xe + ye * ye
        reward = -(pos_err2 + 0.3 * the * the) \
            - self.res_penalty * (a[0] ** 2 + a[1] ** 2)
        terminated = bool(np.sqrt(pos_err2) > 3.0)
        if terminated:
            reward -= 10.0
        truncated = self.step_i >= self.max_steps
        return self._obs, float(reward), terminated, truncated, {}
