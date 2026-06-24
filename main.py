"""Run the WMR slipping trajectory-tracking simulations (paper reproduction).

Examples
--------
    python main.py                 # line + circle, kinematic+adaptive
    python main.py --traj circle
    python main.py --no-adaptive   # baseline without slip compensation
    python main.py --dynamics      # use the dynamic (torque) layer
"""
import argparse

import numpy as np

from wmr.simulate import simulate, plot_results


def rmse(*errs):
    e = np.vstack(errs)
    return float(np.sqrt(np.mean(np.sum(e ** 2, axis=0))))


def run_one(traj, T, dt, dynamics, adaptive, tag):
    log = simulate(traj_name=traj, T=T, dt=dt,
                   use_dynamics=dynamics, adaptive=adaptive)
    files = plot_results(log, traj, tag=tag)
    # steady-state error over the final 20% of the run
    k = int(0.8 * len(log["t"]))
    ss = rmse(log["xe"][k:], log["ye"][k:], log["the"][k:])
    print(f"  [{traj:6s}] tag={tag or 'base':10s} "
          f"steady-state RMSE = {ss:.4e}")
    for f in files:
        print(f"      -> {f}")
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", choices=["line", "circle", "both"], default="both")
    ap.add_argument("--T", type=float, default=60.0)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--dynamics", action="store_true")
    ap.add_argument("--no-adaptive", dest="adaptive", action="store_false")
    args = ap.parse_args()

    trajs = ["line", "circle"] if args.traj == "both" else [args.traj]
    tag = ""
    if args.dynamics:
        tag = "dyn"
    if not args.adaptive:
        tag = (tag + "_noadapt").lstrip("_")

    print("Running WMR slipping trajectory tracking simulation")
    for traj in trajs:
        run_one(traj, args.T, args.dt, args.dynamics, args.adaptive, tag)


if __name__ == "__main__":
    main()
