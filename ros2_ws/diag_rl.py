#!/usr/bin/env python3
"""Diagnostic: run the RL policy on Gazebo for a few seconds, print obs+action."""
import sys, time, math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, 'src/wmr_tb3')
from wmr_tb3.controller_core import pose_error, wrap
from wmr_tb3.trajectories import Circle

hi = np.array([np.inf]*6 + [1.5, 1.5], np.float32)
class _D(gym.Env):
    observation_space = spaces.Box(-hi, hi, dtype=np.float32)
    action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

rclpy.init()
n = Node('diag'); n.set_parameters([Parameter('use_sim_time', value=True)])
st = {'pose': None, 'v': 0.0, 'w': 0.0}
def od(m):
    q = m.pose.pose.orientation
    st['pose'] = (m.pose.pose.position.x, m.pose.pose.position.y,
                  math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z)))
    st['v'] = m.twist.twist.linear.x; st['w'] = m.twist.twist.angular.z
n.create_subscription(Odometry, 'odom', od, 20)
pub = n.create_publisher(Twist, 'cmd_vel', 10)
base = '/home/eugene/STT/rl/models/tb3sac_dr'
vn = VecNormalize.load(base+'_vecnorm.pkl', DummyVecEnv([lambda: _D()])); vn.training=False
m = SAC.load(base)
traj = Circle(R=0.7, wr=0.16)
while st['pose'] is None: rclpy.spin_once(n, timeout_sec=0.05)
p0 = st['pose']; (xr0,yr0,thr0),_,_=traj.state(0.0); dth=wrap(p0[2]-thr0)
t0 = n.get_clock().now().nanoseconds*1e-9
for k in range(160):
    rclpy.spin_once(n, timeout_sec=0.01)
    t = n.get_clock().now().nanoseconds*1e-9 - t0
    (xr,yr,thr),vr,wr = traj.state(t)
    ddx,ddy = xr-xr0, yr-yr0; c,s=math.cos(dth),math.sin(dth)
    ref=(p0[0]+c*ddx-s*ddy, p0[1]+s*ddx+c*ddy, wrap(thr+dth))
    e = pose_error(st['pose'], ref)
    obs = np.array([e[0],e[1],math.sin(e[2]),math.cos(e[2]),vr,wr,st['v']/0.22,st['w']/2.84],np.float32)
    a,_ = m.predict(vn.normalize_obs(obs.reshape(1,-1)), deterministic=True)
    v=float(a[0][0])*0.22; w=float(a[0][1])*2.84
    cmd=Twist(); cmd.linear.x=v; cmd.angular.z=w; pub.publish(cmd)
    if k % 20 == 0:
        print(f't={t:4.1f} e=({e[0]:+.2f},{e[1]:+.2f},{e[2]:+.2f}) vmeas={st["v"]:.3f} -> act v={v:.3f} w={w:.3f}', flush=True)
    time.sleep(0.05)
pub.publish(Twist())
rclpy.shutdown()
