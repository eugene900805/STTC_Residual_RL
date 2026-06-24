#!/usr/bin/env python3
"""Self-contained TB3 slip-tracking experiment (single rclpy process).

Talks to a already-running Gazebo TB3 directly: subscribes /odom (pose), runs the
paper controller, injects the paper's proportional slip model (v_wheel*(1-s),
steps at t=30/50 s), publishes /cmd_vel.  Runs mode='paper' and mode='nocomp'
back-to-back (resetting the sim between) and saves /tmp/tb3_exp.npz.

Avoids multi-node / nohup / remap plumbing entirely.
    python3 run_experiment.py
"""
import sys
import time
import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

sys.path.insert(0, 'src/wmr_tb3')
from wmr_tb3.controller_core import KinematicAdaptiveController, TB3Params, pose_error, wrap
from wmr_tb3.trajectories import Circle, StraightLine

# trajectory selectable from the command line:  python3 run_experiment.py [circle|line]
TRAJ_KIND = sys.argv[1] if len(sys.argv) > 1 else 'circle'
R, WR = 0.7, 0.16          # circle: v = R*WR = 0.112 m/s
LINE_VR = 0.11             # straight line: v = 0.11 m/s (≈ circle speed)
HORIZON = 60.0             # sim seconds
SLIP = dict(sL0=0.08, sR0=0.08, sL1=0.18, sR1=0.15, t_right=30.0, t_left=50.0)
RES_V, RES_W = 0.08, 0.6   # residual-RL action scale (small, on top of base)


def true_slip(t):
    sR = SLIP['sR0'] if t < SLIP['t_right'] else SLIP['sR1']
    sL = SLIP['sL0'] if t < SLIP['t_left'] else SLIP['sL1']
    return sL, sR


class Exp(Node):
    def __init__(self):
        super().__init__('tb3_experiment')
        self.set_parameters([Parameter('use_sim_time', value=True)])
        self.pose = None
        self.v_meas = 0.0
        self.w_meas = 0.0
        self.pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.create_subscription(Odometry, 'odom', self._odom, 20)
        self.reset_cli = self.create_client(Empty, '/reset_simulation')

    def _odom(self, m):
        q = m.pose.pose.orientation
        yaw = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
        self.pose = (m.pose.pose.position.x, m.pose.pose.position.y, yaw)
        self.v_meas = m.twist.twist.linear.x
        self.w_meas = m.twist.twist.angular.z
        # sim time from the odom header (reliable) -- matches the training env's
        # clock source.  get_clock()/`/clock` has an incompatible QoS so it
        # updates slowly, which throttled the control loop to ~5 Hz (!= 20 Hz
        # training rate) and broke the RL policy.
        self._sim_t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9

    def sim_now(self):
        return getattr(self, '_sim_t', 0.0)

    def reset(self):
        if self.reset_cli.wait_for_service(timeout_sec=3.0):
            self.reset_cli.call_async(Empty.Request())
        for _ in range(40):
            rclpy.spin_once(self, timeout_sec=0.05)
        self.pub.publish(Twist())
        time.sleep(0.5)

    def run(self, mode):
        # mode: 'paper' (kinematic adaptive), 'nocomp' (kinematic only),
        #       'veloop' (kinematic + velocity inner loop), 'rl' (SAC policy),
        #       'residual' (velocity inner loop + small learned residual)
        adaptive = (mode == 'paper')
        veloop = (mode == 'veloop')
        rl = (mode == 'rl')
        residual = (mode == 'residual')
        V_MAX, W_MAX = 0.22, 2.84
        rl_model = rl_vn = None
        if rl or residual:
            from stable_baselines3 import SAC
            from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
            import gymnasium as gym
            from gymnasium import spaces
            # residual obs is 10-dim (adds the base command), rl obs is 8-dim
            base = ('/home/eugene/STT/rl/models/tb3sac_gzres' if residual
                    else '/home/eugene/STT/rl/models/tb3sac_gz10')
            ndim = 10 if residual else 8
            hi = np.array([np.inf]*6 + [1.5]*(ndim - 6), np.float32)

            class _Dummy(gym.Env):
                observation_space = spaces.Box(-hi, hi, dtype=np.float32)
                action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
            rl_vn = VecNormalize.load(base + '_vecnorm.pkl',
                                      DummyVecEnv([lambda: _Dummy()]))
            rl_vn.training = False
            rl_model = SAC.load(base)
        # velocity inner-loop PI gains and integrators
        KPV, KIV, KPW, KIW = 0.4, 1.0, 0.4, 1.0
        iv = iw = 0.0
        p = TB3Params()
        ctrl = KinematicAdaptiveController(p, adaptive=adaptive)
        traj = (StraightLine(vr=LINE_VR, alpha=0.0) if TRAJ_KIND == 'line'
                else Circle(R=R, wr=WR))
        # wait for a fresh pose (bounded)
        self.pose = None
        tries = 0
        while self.pose is None and tries < 400:
            rclpy.spin_once(self, timeout_sec=0.05)
            tries += 1
        if self.pose is None:
            raise RuntimeError("no /odom received -- is Gazebo up on this domain?")
        p_start = (self.pose[0], self.pose[1])
        (xr0, yr0, thr0), _, _ = traj.state(0.0)
        dth = wrap(self.pose[2] - thr0)
        ref0 = (xr0, yr0)

        log = {k: [] for k in ['t', 'x', 'y', 'rx', 'ry', 'xe', 'ye', 'the',
                               'sLh', 'sRh', 'sLt', 'sRt']}
        t0 = self.sim_now()
        dt = 0.1   # 10 Hz -- matches the RL training control rate (TB3 rate)
        next_t = time.time()
        while True:
            rclpy.spin_once(self, timeout_sec=0.01)
            t = self.sim_now() - t0
            if t > HORIZON:
                break
            # reference (rigidly anchored to robot start pose)
            (xr, yr, thr), vr, wr = traj.state(t)
            ddx, ddy = xr - ref0[0], yr - ref0[1]
            c, s = math.cos(dth), math.sin(dth)
            rx = p_start[0] + c*ddx - s*ddy
            ry = p_start[1] + s*ddx + c*ddy
            ref = (rx, ry, wrap(thr + dth))
            self._refxy = (rx, ry)

            e = pose_error(self.pose, ref)

            if rl:
                obs = np.array([e[0], e[1], math.sin(e[2]), math.cos(e[2]),
                                vr, wr, self.v_meas/0.22, self.w_meas/2.84],
                               dtype=np.float32)
                nobs = rl_vn.normalize_obs(obs.reshape(1, -1))
                act, _ = rl_model.predict(nobs, deterministic=True)
                v = float(act[0][0]) * V_MAX
                w = float(act[0][1]) * W_MAX
                v = max(min(v, V_MAX), -V_MAX)
                w = max(min(w, W_MAX), -W_MAX)
                self._inject_and_publish(v, w, t, p)
                self._log(log, t, e, ctrl)
                self._wait_sim(dt)
                continue

            vc, wc = ctrl.virtual_velocity(e, vr, wr, self.w_meas)
            ctrl.adapt(e, vc, wc, dt)
            v, w = ctrl.cmd_vel(vc, wc)        # i^-compensated (i=1 if not adaptive)

            if veloop or residual:
                # velocity inner loop: drive measured body velocity -> (vc, wc)
                # (PI; integral compensates persistent slip -- the dynamic layer)
                ev, ew = vc - self.v_meas, wc - self.w_meas
                iv = max(min(iv + ev*dt, 0.5), -0.5)   # anti-windup
                iw = max(min(iw + ew*dt, 1.0), -1.0)
                v = vc + KPV*ev + KIV*iv
                w = wc + KPW*ew + KIW*iw
                v = max(min(v, p.v_max), -p.v_max)
                w = max(min(w, p.w_max), -p.w_max)

            if residual:
                # "algorithm + RL": a small learned residual on top of the base
                # velocity-loop command (obs adds v_base/w_base, 10-dim)
                v_base, w_base = v, w
                obs = np.array([e[0], e[1], math.sin(e[2]), math.cos(e[2]),
                                vr, wr, self.v_meas/V_MAX, self.w_meas/W_MAX,
                                v_base/V_MAX, w_base/W_MAX], dtype=np.float32)
                nobs = rl_vn.normalize_obs(obs.reshape(1, -1))
                act, _ = rl_model.predict(nobs, deterministic=True)
                v = max(min(v_base + float(act[0][0])*RES_V, V_MAX), -V_MAX)
                w = max(min(w_base + float(act[0][1])*RES_W, W_MAX), -W_MAX)

            self._inject_and_publish(v, w, t, p)
            self._log(log, t, e, ctrl)
            self._wait_sim(dt)
        self.pub.publish(Twist())
        return {k: np.array(v) for k, v in log.items()}

    def _wait_sim(self, dt):
        """Pace one control step by SIM time (20 Hz sim, matching training)."""
        tgt = self.sim_now() + dt
        g = 0
        while self.sim_now() < tgt and g < 300:
            rclpy.spin_once(self, timeout_sec=0.01)
            g += 1

    def _inject_and_publish(self, v, w, t, p):
        """Apply the paper proportional slip at the wheel level, publish cmd_vel."""
        sL, sR = true_slip(t)
        wL = (v - p.b*w)/p.r
        wR = (v + p.b*w)/p.r
        vL = p.r*wL*(1-sL)
        vR = p.r*wR*(1-sR)
        cmd = Twist()
        cmd.linear.x = 0.5*(vR+vL)
        cmd.angular.z = (vR-vL)/(2*p.b)
        self.pub.publish(cmd)
        self._slip_now = (sL, sR)

    def _log(self, log, t, e, ctrl):
        sLh, sRh = ctrl.slip_estimate
        sL, sR = getattr(self, '_slip_now', (0.0, 0.0))
        rx, ry = getattr(self, '_refxy', (0.0, 0.0))
        log['t'].append(t); log['x'].append(self.pose[0]); log['y'].append(self.pose[1])
        log['rx'].append(rx); log['ry'].append(ry)
        log['xe'].append(e[0]); log['ye'].append(e[1]); log['the'].append(e[2])
        log['sLh'].append(sLh); log['sRh'].append(sRh)
        log['sLt'].append(sL); log['sRt'].append(sR)


def rmse(L):
    en = np.sqrt(L['xe']**2 + L['ye']**2 + L['the']**2)
    k = int(0.7*len(en))
    return float(np.mean(en[k:]))


def main():
    rclpy.init()
    n = Exp()
    out = {}
    for mode in ['nocomp', 'veloop', 'rl', 'residual']:
        print(f'--- {mode} ---', flush=True)
        n.reset()
        L = n.run(mode)
        out[mode] = L
        print(f'{mode}: samples={len(L["t"])} simT={L["t"][-1]:.1f} '
              f'steady|e|={rmse(L):.4f}  slipEst_last=({L["sLh"][-1]:.2f},{L["sRh"][-1]:.2f})',
              flush=True)
    outnpz = '/tmp/tb3_exp_line.npz' if TRAJ_KIND == 'line' else '/tmp/tb3_exp.npz'
    np.savez(outnpz,
             **{f'{m}_{k}': out[m][k] for m in out for k in out[m]})
    print(f'saved {outnpz}', flush=True)
    n.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
