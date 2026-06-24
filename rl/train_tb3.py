"""Train a SAC policy on the TB3-scaled env (velocity lag + slip).

    python -m rl.train_tb3 --timesteps 250000
"""
import argparse
import os

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from rl.tb3_env import TB3TrackingEnv

MODEL_DIR = "rl/models"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=250_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="tb3sac")
    args = ap.parse_args()
    os.makedirs(MODEL_DIR, exist_ok=True)

    venv = DummyVecEnv([lambda: Monitor(TB3TrackingEnv(seed=args.seed))])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = SAC("MlpPolicy", venv, learning_rate=3e-4, buffer_size=300_000,
                batch_size=256, gamma=0.99, tau=0.005, train_freq=1,
                gradient_steps=1, learning_starts=5_000,
                policy_kwargs=dict(net_arch=[256, 256]), seed=args.seed,
                verbose=1)
    model.learn(total_timesteps=args.timesteps, log_interval=20)

    path = os.path.join(MODEL_DIR, args.tag)
    model.save(path)
    venv.save(path + "_vecnorm.pkl")
    print(f"saved {path}.zip and {path}_vecnorm.pkl")


if __name__ == "__main__":
    main()
