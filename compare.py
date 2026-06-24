"""Overlay tracking-error norm: adaptive slip compensation vs. no compensation.

This single figure summarises the paper's central claim and is the baseline the
upcoming RL controller will be compared against.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wmr.simulate import simulate


def err_norm(log):
    return np.sqrt(log["xe"] ** 2 + log["ye"] ** 2 + log["the"] ** 2)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, traj in zip(axes, ["line", "circle"]):
        on = simulate(traj_name=traj, adaptive=True)
        off = simulate(traj_name=traj, adaptive=False)
        ax.plot(off["t"], err_norm(off), "C3", label="no compensation")
        ax.plot(on["t"], err_norm(on), "C0", label="adaptive (paper)")
        for tc in (30.0, 50.0):
            ax.axvline(tc, color="k", ls=":", lw=0.8)
        ax.set_xlabel("t [s]")
        ax.set_ylabel(r"$\|e\|$")
        ax.set_title(f"{traj}: slip steps at t=30s, 50s")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle("Tracking error: adaptive slip compensation vs. none")
    fig.tight_layout()
    out = "figures/compare_adaptive.png"
    fig.savefig(out, dpi=130)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
