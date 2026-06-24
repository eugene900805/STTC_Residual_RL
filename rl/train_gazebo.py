"""Train SAC INSIDE Gazebo (physics-in-the-loop) -- no sim-to-real gap.

Requires a single headless Gazebo (empty_fast.world, RTF~3) with a spawned TB3
already running.  Each env step advances the real Gazebo physics.

    python -m rl.train_gazebo --timesteps 150000
"""
import argparse
import os

import rclpy
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

from rl.gazebo_env import GazeboTB3Env

MODEL_DIR = "rl/models"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=150_000)
    ap.add_argument("--tag", default="tb3sac_gz")
    ap.add_argument("--residual", action="store_true",
                    help="train an RL residual on top of the velocity-loop base")
    args = ap.parse_args()
    os.makedirs(MODEL_DIR, exist_ok=True)

    if args.residual:
        from rl.gazebo_residual_env import GazeboResidualEnv
        env = GazeboResidualEnv(horizon=20.0, traj="mix", seed=0)
    else:
        env = GazeboTB3Env(horizon=20.0, traj="mix", seed=0)
    venv = DummyVecEnv([lambda: env])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # device='cpu': SAC's net is tiny and the bottleneck is Gazebo stepping, not
    # the network -- and CUDA + rclpy spinning in one process can segfault.
    model = SAC("MlpPolicy", venv, learning_rate=3e-4, buffer_size=100_000,
                batch_size=256, gamma=0.99, tau=0.005, train_freq=1,
                gradient_steps=1, learning_starts=3_000, device="cpu",
                policy_kwargs=dict(net_arch=[256, 256]), seed=0, verbose=1)
    # checkpoint periodically so a long Gazebo run survives a crash
    ckpt = CheckpointCallback(save_freq=20_000, save_path=MODEL_DIR,
                              name_prefix=args.tag + "_ckpt",
                              save_vecnormalize=True)
    model.learn(total_timesteps=args.timesteps, log_interval=10, callback=ckpt)

    path = os.path.join(MODEL_DIR, args.tag)
    model.save(path)
    venv.save(path + "_vecnorm.pkl")
    print(f"saved {path}.zip and {path}_vecnorm.pkl", flush=True)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
