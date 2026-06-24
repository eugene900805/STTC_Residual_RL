"""Evaluate a trained SAC policy on the paper's exact scenarios and compare it
head-to-head with the paper's kinematic + adaptive backstepping controller.

    python -m rl.evaluate --tag sac --traj mix

Produces, for the straight-line and circle references (slip stepping at
t=30s / t=50s, same initial offset as the paper simulation):
  * steady-state RMSE table: RL vs. paper controller
  * overlay figures: trajectory, and tracking-error norm vs. time
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from rl.algos import load_model, ALGOS
from rl.wmr_env import WMRTrackingEnv
from rl.residual_env import WMRResidualEnv
from wmr.model import SlipProfile
from wmr.controller import pose_error
from wmr.simulate import simulate as paper_simulate

MODEL_DIR = "rl/models"
OUTDIR = "figures_rl"

PAPER_POSE0 = {"line": (1.6, 0.0, np.pi / 4), "circle": (1.5, -0.4, 0.0)}


def steady_rmse(t, xe, ye, the):
    k = int(0.8 * len(t))
    return float(np.sqrt(np.mean(xe[k:] ** 2 + ye[k:] ** 2 + the[k:] ** 2)))


def rollout_rl(model_path, vecnorm_path, traj, horizon=60.0, dt=0.05,
               residual=False, base_adaptive=True, algo="sac"):
    # raw env so there is no VecEnv auto-reset corrupting the final samples;
    # observations are normalized manually with the saved VecNormalize stats.
    if residual:
        env = WMRResidualEnv(traj=traj, randomize=False, horizon=horizon, dt=dt,
                             slip_profile=SlipProfile(), pose0=PAPER_POSE0[traj],
                             base_adaptive=base_adaptive)
    else:
        env = WMRTrackingEnv(traj=traj, randomize=False, horizon=horizon, dt=dt,
                             slip_profile=SlipProfile(), pose0=PAPER_POSE0[traj])
    vn = VecNormalize.load(vecnorm_path, DummyVecEnv([lambda: env]))
    vn.training = False
    model = load_model(algo, model_path)

    obs, _ = env.reset()
    log = {k: [] for k in ["t", "x", "y", "xr", "yr", "xe", "ye", "the"]}
    done = False
    while not done:
        nobs = vn.normalize_obs(obs.reshape(1, -1).astype(np.float32))
        action, _ = model.predict(nobs, deterministic=True)
        obs, _, term, trunc, _ = env.step(action[0])
        done = term or trunc
        ref, _, _ = env.traj.state(env.t)
        e = pose_error(env.plant.pose, ref)
        log["t"].append(env.t)
        log["x"].append(env.plant.pose[0]); log["y"].append(env.plant.pose[1])
        log["xr"].append(ref[0]); log["yr"].append(ref[1])
        log["xe"].append(e[0]); log["ye"].append(e[1]); log["the"].append(e[2])
    return {k: np.array(v) for k, v in log.items()}


def plot_compare(rl, paper, traj, label, fname):
    os.makedirs(OUTDIR, exist_ok=True)
    # trajectory overlay
    fig, ax = plt.subplots(figsize=(6, 5))
    # trajectories first, reference drawn LAST (on top) so it stays visible
    # where the policy hugs it; thick black dashes show over the colored lines.
    ax.plot(paper["x"], paper["y"], "C2-", lw=1.6, label="paper controller", zorder=2)
    ax.plot(rl["x"], rl["y"], "C0-", lw=1.6, label=label, zorder=3)
    ax.plot(rl["xr"], rl["yr"], "k--", lw=2.2, dashes=(6, 4),
            label="reference", zorder=5)
    ax.plot(rl["x"][0], rl["y"][0], "ro", label="start", zorder=6)
    ax.axis("equal"); ax.grid(True, alpha=0.3); ax.legend()
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
    ax.set_title(f"{traj}: {label} vs. paper controller")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, f"{traj}_traj_{fname}.png"), dpi=130)
    plt.close(fig)

    # error-norm overlay
    en_rl = np.sqrt(rl["xe"] ** 2 + rl["ye"] ** 2 + rl["the"] ** 2)
    en_pp = np.sqrt(paper["xe"] ** 2 + paper["ye"] ** 2 + paper["the"] ** 2)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(paper["t"], en_pp, "C2", lw=1.3, label="paper controller")
    ax.plot(rl["t"], en_rl, "C0", lw=1.3, label=label)
    for tc in (30.0, 50.0):
        ax.axvline(tc, color="k", ls=":", lw=0.8)
    ax.set_xlabel("t [s]"); ax.set_ylabel(r"$\|e\|$")
    ax.set_title(f"{traj}: tracking-error norm (slip steps at 30s, 50s)")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, f"{traj}_error_{fname}.png"), dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default="sac", choices=list(ALGOS),
                    help="RL algorithm of the model being evaluated")
    ap.add_argument("--tag", default=None,
                    help="model name prefix (defaults to the algo name)")
    ap.add_argument("--traj", default="mix", choices=["line", "circle", "mix"],
                    help="which trained model to load")
    ap.add_argument("--residual", action="store_true",
                    help="evaluate a residual-RL model (base = paper controller)")
    ap.add_argument("--no-base-adapt", dest="base_adaptive",
                    action="store_false",
                    help="residual base uses NO slip compensation (option B)")
    args = ap.parse_args()
    tag = args.tag or args.algo

    model_path = os.path.join(MODEL_DIR, f"{tag}_{args.traj}")
    vecnorm_path = model_path + "_vecnorm.pkl"
    algo_lbl = args.algo.upper()
    label = f"residual RL ({algo_lbl})" if args.residual else f"RL ({algo_lbl})"
    fname = f"res_{tag}_vs_paper" if args.residual else f"{tag}_vs_paper"

    print(f"Evaluating {label}: {model_path} vs. paper controller\n")
    print(f"{'traj':8s} {label+' RMSE':>16s} {'paper RMSE':>12s}")
    for traj in ["line", "circle"]:
        rl = rollout_rl(model_path, vecnorm_path, traj, residual=args.residual,
                        base_adaptive=args.base_adaptive, algo=args.algo)
        pp = paper_simulate(traj_name=traj, T=60.0, dt=0.01,
                            pose0=PAPER_POSE0[traj], use_dynamics=False,
                            adaptive=True)
        rl_rmse = steady_rmse(rl["t"], rl["xe"], rl["ye"], rl["the"])
        pp_rmse = steady_rmse(pp["t"], pp["xe"], pp["ye"], pp["the"])
        print(f"{traj:8s} {rl_rmse:16.4e} {pp_rmse:12.4e}")
        plot_compare(rl, pp, traj, label, fname)
    print(f"\nFigures saved in ./{OUTDIR}/")


if __name__ == "__main__":
    main()
