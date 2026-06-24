"""Deploy residual-RL (algorithm + RL) on TB3 in Gazebo and compare to the base
velocity-loop controller, on the paper scenario (circle, slip steps 30/50 s)."""
import sys
import numpy as np
import rclpy
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl.gazebo_residual_env import GazeboResidualEnv

sys.path.insert(0, '/home/eugene/STT/ros2_ws/src/wmr_tb3')
from wmr_tb3.trajectories import Circle

BASE = '/home/eugene/STT/rl/models/tb3sac_gzres'


def paper_slip(t):
    sR = 0.08 if t < 30.0 else 0.15
    sL = 0.08 if t < 50.0 else 0.18
    return sL, sR


def rollout(env, model, vn):
    o, _ = env.reset()
    es = []
    done = False
    while not done:
        if model is None:
            a = np.zeros(2)                  # base velocity-loop only
        else:
            no = vn.normalize_obs(o.reshape(1, -1).astype(np.float32))
            a, _ = model.predict(no, deterministic=True)
            a = a[0]
        o, r, term, trunc, _ = env.step(a)
        done = term or trunc
        es.append(list(env._e) + [env.pose[0], env.pose[1]])
    return np.array(es)


def rmse(es):
    en = np.sqrt(es[:, 0]**2 + es[:, 1]**2 + es[:, 2]**2)
    k = int(0.7 * len(en))
    return float(np.mean(en[k:]))


def main():
    env = GazeboResidualEnv(horizon=60.0,
                            fixed_traj=Circle(R=0.7, wr=0.16),
                            fixed_slip=paper_slip)
    vn = VecNormalize.load(BASE + '_vecnorm.pkl', DummyVecEnv([lambda: env]))
    vn.training = False
    model = SAC.load(BASE)

    base = rollout(env, None, None)
    print('base (velocity loop) steady RMSE = %.4f' % rmse(base), flush=True)
    res = rollout(env, model, vn)
    print('base + RL residual  steady RMSE = %.4f' % rmse(res), flush=True)

    np.savez('/tmp/tb3_residual.npz', base=base, res=res)
    print('saved /tmp/tb3_residual.npz', flush=True)
    env.close()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
