"""Gazebo-in-the-loop gymnasium env for TB3 slip tracking (physics-in-the-loop).

Each RL step actually advances the Gazebo TB3 simulation, so the training
dynamics == the deployment dynamics -> no sim-to-real gap.  The paper
proportional slip is injected at the wheel level (same as deployment).  Gazebo
must be running headless with real_time_update_rate=0 (empty_fast.world) so it
runs as fast as the CPU allows; steps are paced by /clock, not wall time.

Observation (8): [xe, ye, sin th_e, cos th_e, vr, wr, v/V_MAX, w/W_MAX]
Action (2):      [v, w] in [-1,1] -> [+-V_MAX, +-W_MAX]
"""
import math
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from rosgraph_msgs.msg import Clock
from std_srvs.srv import Empty

import sys
sys.path.insert(0, '/home/eugene/STT/ros2_ws/src/wmr_tb3')
from wmr_tb3.controller_core import pose_error, wrap
from wmr_tb3.trajectories import Circle, StraightLine

R_W, B_W = 0.033, 0.08
V_MAX, W_MAX = 0.22, 2.84


class GazeboTB3Env(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, dt=0.1, horizon=20.0, traj="mix",
                 fixed_traj=None, fixed_slip=None, seed=None):
        super().__init__()
        if not rclpy.ok():
            rclpy.init()
        self.node = Node('gazebo_rl_env')
        self.fixed_traj = fixed_traj   # for deployment: a fixed Circle/Line
        self.fixed_slip = fixed_slip   # for deployment: a fixed slip(t) fn
        self.dt = dt        # control period [s] = 1/TB3 control rate (10 Hz)
        self.horizon = horizon
        self.max_steps = int(round(horizon / dt))
        self.traj_mode = traj
        self.rng = np.random.default_rng(seed)

        self.pose = None
        self.v = 0.0
        self.w = 0.0
        self.sim_t = 0.0

        self.pub = self.node.create_publisher(Twist, 'cmd_vel', 10)
        self.node.create_subscription(Odometry, 'odom', self._odom, 20)
        self.node.create_subscription(Clock, '/clock', self._clock, 20)
        self.reset_cli = self.node.create_client(Empty, '/reset_world')
        # pause/unpause physics -> each control step is EXACTLY dt of sim time,
        # independent of compute time or real-time factor (deterministic 10 Hz).
        self.pause_cli = self.node.create_client(Empty, '/pause_physics')
        self.unpause_cli = self.node.create_client(Empty, '/unpause_physics')

        high = np.array([np.inf]*6 + [1.5, 1.5], np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    # ------------------------------------------------------------------ #
    def _odom(self, m):
        q = m.pose.pose.orientation
        self.pose = (m.pose.pose.position.x, m.pose.pose.position.y,
                     math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z)))
        self.v = m.twist.twist.linear.x
        self.w = m.twist.twist.angular.z
        # sim time from the odom header (robust vs /clock QoS issues)
        self.sim_t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9

    def _clock(self, m):
        self.sim_t = m.clock.sec + m.clock.nanosec * 1e-9

    def _spin(self, n=1):
        for _ in range(n):
            rclpy.spin_once(self.node, timeout_sec=0.02)

    def _srv(self, cli):
        if cli.wait_for_service(timeout_sec=2.0):
            fut = cli.call_async(Empty.Request())
            for _ in range(50):
                rclpy.spin_once(self.node, timeout_sec=0.01)
                if fut.done():
                    return

    # ------------------------------------------------------------------ #
    def _make_traj(self):
        if self.traj_mode == "circle" or (self.traj_mode == "mix" and self.rng.random() < 0.5):
            return Circle(R=self.rng.uniform(0.5, 0.9), wr=self.rng.uniform(0.14, 0.20))
        return StraightLine(vr=self.rng.uniform(0.08, 0.13), alpha=0.0)

    def _sample_slip(self):
        sL0, sR0 = self.rng.uniform(0.0, 0.15, size=2)
        sL1, sR1 = self.rng.uniform(0.0, 0.22, size=2)
        tL = self.rng.uniform(0.3, 0.7) * self.horizon
        tR = self.rng.uniform(0.3, 0.7) * self.horizon

        def slip(t):
            return (sL0 if t < tL else sL1), (sR0 if t < tR else sR1)
        return slip

    # ------------------------------------------------------------------ #
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.pub.publish(Twist())
        self._srv(self.unpause_cli)   # physics must run during reset/settle
        # reset model poses (keeps sim clock monotonic)
        if self.reset_cli.wait_for_service(timeout_sec=5.0):
            fut = self.reset_cli.call_async(Empty.Request())
            t0 = self.sim_t
            for _ in range(60):
                self._spin()
                if fut.done():
                    break
        # let the robot settle at spawn
        for _ in range(20):
            self.pub.publish(Twist())
            self._spin()
        tries = 0
        while self.pose is None and tries < 300:
            self._spin()
            tries += 1

        self.traj = self.fixed_traj or self._make_traj()
        self.slip = self.fixed_slip or self._sample_slip()
        (xr0, yr0, thr0), _, _ = self.traj.state(0.0)
        self.p_start = (self.pose[0], self.pose[1])
        self.ref0 = (xr0, yr0)
        self.dth = wrap(self.pose[2] - thr0)
        self.t0 = self.sim_t
        self.step_i = 0
        self._srv(self.pause_cli)   # freeze sim; step() unpauses for exactly dt
        return self._obs(), {}

    def _ref(self, t):
        (xr, yr, thr), vr, wr = self.traj.state(t)
        ddx, ddy = xr - self.ref0[0], yr - self.ref0[1]
        c, s = math.cos(self.dth), math.sin(self.dth)
        return ((self.p_start[0] + c*ddx - s*ddy,
                 self.p_start[1] + s*ddx + c*ddy, wrap(thr + self.dth)), vr, wr)

    def _obs(self):
        t = self.sim_t - self.t0
        ref, vr, wr = self._ref(t)
        e = pose_error(self.pose, ref)
        self._e, self._vr, self._wr = e, vr, wr
        return np.array([e[0], e[1], math.sin(e[2]), math.cos(e[2]), vr, wr,
                         self.v / V_MAX, self.w / W_MAX], dtype=np.float32)

    def step(self, action):
        a = np.clip(action, -1.0, 1.0)
        v = float(a[0]) * V_MAX
        w = float(a[1]) * W_MAX
        t = self.sim_t - self.t0
        sL, sR = self.slip(t)
        wL = (v - B_W * w) / R_W
        wR = (v + B_W * w) / R_W
        vL = R_W * wL * (1.0 - sL)
        vR = R_W * wR * (1.0 - sR)
        cmd = Twist()
        cmd.linear.x = 0.5 * (vR + vL)
        cmd.angular.z = (vR - vL) / (2.0 * B_W)

        # advance Gazebo by EXACTLY dt of sim time: publish cmd, unpause physics,
        # let sim run dt, pause.  Compute (obs/reward + the agent's next action)
        # then happens while paused -> sim frozen -> deterministic 10 Hz control.
        self.pub.publish(cmd)
        self._srv(self.unpause_cli)
        target = self.sim_t + self.dt
        guard = 0
        while self.sim_t < target and guard < 400:
            rclpy.spin_once(self.node, timeout_sec=0.005)
            guard += 1
        self._srv(self.pause_cli)

        self._nstep = getattr(self, '_nstep', 0) + 1
        if self._nstep % 500 == 0:
            print(f"[env] step={self._nstep} sim_t={self.sim_t:.1f} "
                  f"pose=({self.pose[0]:.2f},{self.pose[1]:.2f})",
                  file=sys.stderr, flush=True)

        obs = self._obs()
        xe, ye, the = self._e
        pos2 = xe*xe + ye*ye
        reward = -(pos2 + 0.3*the*the) - 1e-3*(a[0]**2 + a[1]**2)
        self.step_i += 1
        terminated = bool(np.sqrt(pos2) > 1.5)
        if terminated:
            reward -= 10.0
        truncated = self.step_i >= self.max_steps
        return obs, float(reward), terminated, truncated, {}

    def close(self):
        try:
            self.pub.publish(Twist())
            self.node.destroy_node()
        except Exception:
            pass
