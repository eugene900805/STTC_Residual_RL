"""TB3-scaled RL env with DOMAIN RANDOMIZATION for sim-to-real transfer.

A naive policy trained on a single fixed dynamics model fails on the real
Gazebo TB3 (it spirals to the centre).  To close that gap:

  * observation includes the robot's current body velocity (v, w) so the policy
    can cope with velocity-tracking dynamics (lag/accel limits);
  * every episode randomises the dynamics: velocity time-constant tau,
    acceleration limits, an action delay, per-wheel scale error, process and
    observation noise -- plus the paper proportional slip.

Observation (8): [xe, ye, sin th_e, cos th_e, vr, wr, v/V_MAX, w/W_MAX]
Action (2):      [v, w] in [-1,1] -> [+-V_MAX, +-W_MAX]
"""
import collections
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from wmr.controller import pose_error
from wmr.trajectories import Circle, StraightLine

R_W, B_W = 0.033, 0.08
V_MAX, W_MAX = 0.22, 2.84


class TB3TrackingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, dt=0.05, horizon=30.0, randomize=True, traj="mix",
                 domain_rand=True, slip=None, pose0=None, tau=0.25, seed=None):
        super().__init__()
        self.dt = dt
        self.horizon = horizon
        self.max_steps = int(round(horizon / dt))
        self.randomize = randomize
        self.domain_rand = domain_rand
        self.traj_mode = traj
        self.fixed_slip = slip
        self.fixed_pose0 = pose0
        self.tau_fixed = tau
        self.rng = np.random.default_rng(seed)

        high = np.array([np.inf]*6 + [1.5, 1.5], np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    def _make_traj(self):
        if self.traj_mode == "circle" or (self.traj_mode == "mix" and self.rng.random() < 0.5):
            Rr = self.rng.uniform(0.5, 0.9) if self.randomize else 0.7
            wr = self.rng.uniform(0.14, 0.20) if self.randomize else 0.18
            return Circle(R=Rr, wr=wr)
        vr = self.rng.uniform(0.08, 0.13) if self.randomize else 0.12
        return StraightLine(vr=vr, alpha=0.0)

    def _sample_slip(self):
        sL0, sR0 = self.rng.uniform(0.0, 0.15, size=2)
        sL1, sR1 = self.rng.uniform(0.0, 0.22, size=2)
        tL = self.rng.uniform(0.3, 0.7) * self.horizon
        tR = self.rng.uniform(0.3, 0.7) * self.horizon

        def slip(t):
            return (sL0 if t < tL else sL1), (sR0 if t < tR else sR1)
        return slip

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.traj = self._make_traj()
        self.slip = self.fixed_slip or (
            self._sample_slip() if self.randomize else (lambda t: (0.1, 0.1)))

        # --- domain randomisation of the dynamics ---
        if self.domain_rand:
            self.tau = self.rng.uniform(0.10, 0.55)
            self.acc_v = self.rng.uniform(0.3, 1.2)      # m/s^2
            self.acc_w = self.rng.uniform(2.0, 8.0)      # rad/s^2
            self.delay = int(self.rng.integers(0, 3))    # action delay [steps]
            self.scaleL = self.rng.uniform(0.9, 1.1)     # per-wheel scale error
            self.scaleR = self.rng.uniform(0.9, 1.1)
            self.proc = self.rng.uniform(0.0, 0.01)      # process noise
            self.obsn = self.rng.uniform(0.0, 0.01)      # obs noise
        else:
            self.tau, self.acc_v, self.acc_w = self.tau_fixed, 1.0, 6.0
            self.delay = 0
            self.scaleL = self.scaleR = 1.0
            self.proc = self.obsn = 0.0
        self.abuf = collections.deque([np.zeros(2)]*(self.delay+1),
                                      maxlen=self.delay+1)

        ref0, _, _ = self.traj.state(0.0)
        if self.fixed_pose0 is not None:
            self.pose = np.array(self.fixed_pose0, float)
        elif self.randomize:
            off = self.rng.uniform(-0.25, 0.25, size=3)
            off[2] = self.rng.uniform(-0.5, 0.5)
            self.pose = ref0 + off
        else:
            self.pose = ref0 + np.array([-0.15, -0.15, 0.3])
        self.v = 0.0
        self.w = 0.0
        self.t = 0.0
        self.step_i = 0
        return self._obs(), {}

    def _obs(self):
        ref, vr, wr = self.traj.state(self.t)
        e = pose_error(self.pose, ref)
        self._e = e
        n = self.obsn
        return np.array([
            e[0] + n*self.rng.standard_normal(),
            e[1] + n*self.rng.standard_normal(),
            np.sin(e[2]), np.cos(e[2]), vr, wr,
            self.v / V_MAX, self.w / W_MAX], dtype=np.float32)

    def step(self, action):
        self.abuf.append(np.clip(action, -1.0, 1.0))
        a = self.abuf[0]                       # delayed action
        cmd_v = float(a[0]) * V_MAX
        cmd_w = float(a[1]) * W_MAX

        # paper slip (with per-wheel scale error) -> slipped body target
        sL, sR = self.slip(self.t)
        wL = (cmd_v - B_W * cmd_w) / R_W
        wR = (cmd_v + B_W * cmd_w) / R_W
        vL = R_W * wL * (1.0 - sL) * self.scaleL
        vR = R_W * wR * (1.0 - sR) * self.scaleR
        v_t = 0.5 * (vR + vL)
        w_t = (vR - vL) / (2.0 * B_W)

        # first-order lag + acceleration limit + process noise
        a_lag = self.dt / (self.tau + self.dt)
        dv = a_lag * (v_t - self.v)
        dw = a_lag * (w_t - self.w)
        dv = np.clip(dv, -self.acc_v*self.dt, self.acc_v*self.dt)
        dw = np.clip(dw, -self.acc_w*self.dt, self.acc_w*self.dt)
        self.v += dv + self.proc*self.rng.standard_normal()
        self.w += dw + self.proc*self.rng.standard_normal()

        th = self.pose[2]
        self.pose = self.pose + self.dt * np.array(
            [self.v*np.cos(th), self.v*np.sin(th), self.w])
        self.t += self.dt
        self.step_i += 1

        obs = self._obs()
        xe, ye, the = self._e
        pos2 = xe*xe + ye*ye
        reward = -(pos2 + 0.3*the*the) - 1e-3*(a[0]**2 + a[1]**2)
        terminated = bool(np.sqrt(pos2) > 1.5)
        if terminated:
            reward -= 10.0
        truncated = self.step_i >= self.max_steps
        return obs, float(reward), terminated, truncated, {}
