"""Train an RL policy (SAC / PPO / TD3 / DDPG) on the WMR slipping tracking
environment. All algorithms share the same observation, action, network size
and training budget so they can be compared head-to-head.

    python -m rl.train --algo sac  --timesteps 300000
    python -m rl.train --algo ppo  --timesteps 300000
    python -m rl.train --algo td3  --timesteps 300000
    python -m rl.train --algo ddpg --timesteps 300000
"""
import argparse
import os

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from rl.algos import make_model, ALGOS
from rl.wmr_env import WMRTrackingEnv
from rl.residual_env import WMRResidualEnv

MODEL_DIR = "rl/models"


def make_env(traj="mix", seed=0, residual=False, base_adaptive=True):
    def _f():
        if residual:
            env = WMRResidualEnv(traj=traj, randomize=True,
                                 base_adaptive=base_adaptive, seed=seed)
        else:
            env = WMRTrackingEnv(traj=traj, randomize=True, seed=seed)
        return Monitor(env)
    return _f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--traj", default="mix", choices=["line", "circle", "mix"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--algo", default="sac", choices=list(ALGOS),
                    help="RL algorithm: sac | ppo | td3 | ddpg")
    ap.add_argument("--tag", default=None,
                    help="model name prefix (defaults to the algo name)")
    ap.add_argument("--residual", action="store_true",
                    help="train a residual on top of the paper controller")
    ap.add_argument("--no-base-adapt", dest="base_adaptive",
                    action="store_false",
                    help="residual base uses NO slip compensation (option B)")
    args = ap.parse_args()
    tag = args.tag or args.algo

    os.makedirs(MODEL_DIR, exist_ok=True)

    venv = DummyVecEnv([make_env(args.traj, args.seed, args.residual,
                                 args.base_adaptive)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = make_model(args.algo, venv, seed=args.seed)
    print(f"Training {args.algo.upper()} for {args.timesteps} steps "
          f"(traj={args.traj}, tag={tag})")
    model.learn(total_timesteps=args.timesteps, progress_bar=False, log_interval=20)

    model_path = os.path.join(MODEL_DIR, f"{tag}_{args.traj}")
    model.save(model_path)
    venv.save(os.path.join(MODEL_DIR, f"{tag}_{args.traj}_vecnorm.pkl"))
    print(f"saved model -> {model_path}.zip")
    print(f"saved vecnorm stats -> {model_path}_vecnorm.pkl")


if __name__ == "__main__":
    main()
