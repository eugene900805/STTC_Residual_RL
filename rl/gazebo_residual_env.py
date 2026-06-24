"""Residual-RL on top of the model-based controller, trained in Gazebo.

base = paper kinematic backstepping + velocity inner loop (the controller that
already wins on the real TB3, ~0.027 m).  The RL agent only outputs a SMALL
residual (dv, dw) added to the base command -- "algorithm + RL".  Physics-in-the
-loop + deterministic 10 Hz (pause/unpause), inheriting GazeboTB3Env.

Observation (10): [xe, ye, sin th_e, cos th_e, vr, wr, v/V_MAX, w/W_MAX,
                   v_base/V_MAX, w_base/W_MAX]
Action (2):       [dv, dw] in [-1,1] -> [+-RES_V, +-RES_W]  (small)
"""
import math
import numpy as np
from gymnasium import spaces

from rl.gazebo_env import GazeboTB3Env, V_MAX, W_MAX, R_W, B_W

import sys
sys.path.insert(0, '/home/eugene/STT/ros2_ws/src/wmr_tb3')
from wmr_tb3.controller_core import KinematicAdaptiveController, TB3Params, pose_error

RES_V = 0.08    # max residual linear vel [m/s]  (vs 0.22 limit)
RES_W = 0.6     # max residual angular vel [rad/s]
# velocity inner-loop PI gains (same as the deployed veloop)
KPV, KIV, KPW, KIW = 0.4, 1.0, 0.4, 1.0


class GazeboResidualEnv(GazeboTB3Env):
    def __init__(self, **kw):
        super().__init__(**kw)
        high = np.array([np.inf]*6 + [1.5, 1.5, 1.5, 1.5], np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        # init the base controller (kinematic, NO adaptive -> the velocity loop
        # rejects slip) and the PI integrators
        self.ctrl = KinematicAdaptiveController(TB3Params, adaptive=False)
        self.iv = 0.0
        self.iw = 0.0
        return super().reset(seed=seed, options=options)

    def _base_cmd(self):
        """Paper kinematic law + velocity inner loop -> (v_base, w_base)."""
        t = self.sim_t - self.t0
        ref, vr, wr = self._ref(t)
        e = pose_error(self.pose, ref)
        vc, wc = self.ctrl.virtual_velocity(e, vr, wr, self.w)
        ev, ew = vc - self.v, wc - self.w
        self.iv = float(np.clip(self.iv + ev*self.dt, -0.5, 0.5))
        self.iw = float(np.clip(self.iw + ew*self.dt, -1.0, 1.0))
        v_base = vc + KPV*ev + KIV*self.iv
        w_base = wc + KPW*ew + KIW*self.iw
        v_base = float(np.clip(v_base, -V_MAX, V_MAX))
        w_base = float(np.clip(w_base, -W_MAX, W_MAX))
        self._vbase, self._wbase = v_base, w_base
        return v_base, w_base

    def _obs(self):
        # base 8-dim obs from parent, plus the base command
        o = super()._obs()
        vb = getattr(self, '_vbase', 0.0)
        wb = getattr(self, '_wbase', 0.0)
        return np.concatenate([o, [vb / V_MAX, wb / W_MAX]]).astype(np.float32)

    def step(self, action):
        a = np.clip(action, -1.0, 1.0)
        v_base, w_base = self._base_cmd()
        v = float(np.clip(v_base + a[0]*RES_V, -V_MAX, V_MAX))
        w = float(np.clip(w_base + a[1]*RES_W, -W_MAX, W_MAX))

        # paper slip at wheel level -> body cmd, then advance Gazebo exactly dt
        sL, sR = self.slip(self.sim_t - self.t0)
        wL = (v - B_W*w) / R_W
        wR = (v + B_W*w) / R_W
        from geometry_msgs.msg import Twist
        cmd = Twist()
        cmd.linear.x = 0.5*(R_W*wL*(1-sL) + R_W*wR*(1-sR))
        cmd.angular.z = (R_W*wR*(1-sR) - R_W*wL*(1-sL)) / (2*B_W)
        self.pub.publish(cmd)
        self._srv(self.unpause_cli)
        import rclpy
        target = self.sim_t + self.dt
        g = 0
        while self.sim_t < target and g < 400:
            rclpy.spin_once(self.node, timeout_sec=0.005)
            g += 1
        self._srv(self.pause_cli)

        obs = self._obs()
        xe, ye, the = self._e
        pos2 = xe*xe + ye*ye
        # reward: tracking error + small penalty so the residual only helps
        reward = -(pos2 + 0.3*the*the) - 0.02*(a[0]**2 + a[1]**2)
        self.step_i += 1
        terminated = bool(np.sqrt(pos2) > 1.5)
        if terminated:
            reward -= 10.0
        truncated = self.step_i >= self.max_steps
        return obs, float(reward), terminated, truncated, {}
