"""Regenerate the model-based result figures cited by the paper in a clean,
consistent, serif, no-internal-title style (the description lives in the Word
caption).  Run after the simulator is in place:

    python3 -m paper.make_figures

Regenerates, into figures/ and figures_rl/:
  compare_adaptive.png   line_slip.png   circle_trajectory.png   algo_compare.png

The reinforcement-learning trajectory plots and the TurtleBot3/Gazebo plots are
left untouched because they depend on trained policies / recorded Gazebo runs.
"""
import os
import numpy as np

from paper import figstyle
figstyle.apply()
import matplotlib.pyplot as plt

from wmr.simulate import simulate

ROOT = os.path.dirname(os.path.dirname(__file__))
FIG = os.path.join(ROOT, "figures")
FIG_RL = os.path.join(ROOT, "figures_rl")
SLIP_STEPS = (30.0, 50.0)


def _err_norm(log):
    return np.sqrt(log["xe"] ** 2 + log["ye"] ** 2 + log["the"] ** 2)


def compare_adaptive():
    """Tracking-error norm, adaptive slip compensation vs. none (line + circle)."""
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8), sharey=False)
    for ax, traj, ttl in zip(axes, ("line", "circle"), ("(a) line", "(b) circle")):
        on = simulate(traj_name=traj, adaptive=True)
        off = simulate(traj_name=traj, adaptive=False)
        ax.plot(off["t"], _err_norm(off), color=figstyle.COLORS["DDPG"],
                lw=1.6, label="no compensation")
        ax.plot(on["t"], _err_norm(on), color=figstyle.COLORS["SAC"],
                lw=1.8, label="adaptive (paper)")
        for tc in SLIP_STEPS:
            ax.axvline(tc, color="0.5", ls=(0, (2, 2)), lw=0.8)
        ax.set_xlabel("time  $t$ [s]")
        ax.set_ylabel(r"error norm  $\|e\|$")
        ax.set_title(ttl, fontsize=9)
        figstyle.finish(ax)
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    p = os.path.join(FIG, "compare_adaptive.png")
    fig.savefig(p)
    plt.close(fig)
    print("saved", p)


def line_slip():
    """Online slip-ratio estimation on the straight line."""
    log = simulate(traj_name="line", adaptive=True)
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.plot(log["t"], log["sL_true"], color=figstyle.COLORS["SAC"],
            ls=(0, (4, 2)), lw=1.4, label=r"$s_L$ (true)")
    ax.plot(log["t"], log["sL_hat"], color=figstyle.COLORS["SAC"], lw=1.8,
            label=r"$\hat{s}_L$")
    ax.plot(log["t"], log["sR_true"], color=figstyle.COLORS["DDPG"],
            ls=(0, (4, 2)), lw=1.4, label=r"$s_R$ (true)")
    ax.plot(log["t"], log["sR_hat"], color=figstyle.COLORS["DDPG"], lw=1.8,
            label=r"$\hat{s}_R$")
    for tc in SLIP_STEPS:
        ax.axvline(tc, color="0.5", ls=(0, (2, 2)), lw=0.8)
    ax.set_xlabel("time  $t$ [s]")
    ax.set_ylabel("slip ratio")
    ax.legend(ncol=2, fontsize=7.5, loc="upper left")
    figstyle.finish(ax)
    fig.tight_layout()
    p = os.path.join(FIG, "line_slip.png")
    fig.savefig(p)
    plt.close(fig)
    print("saved", p)


def circle_trajectory():
    """Circular-path tracking of the model-based controller (abstract sim)."""
    log = simulate(traj_name="circle", adaptive=True)
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    ax.plot(log["xr"], log["yr"], color=figstyle.COLORS["ref"],
            ls=(0, (5, 3)), lw=1.6, label="reference")
    ax.plot(log["x"], log["y"], color=figstyle.COLORS["SAC"], lw=1.8,
            label="controller")
    ax.plot(log["x"][0], log["y"][0], "o", color=figstyle.COLORS["DDPG"],
            ms=6, label="start")
    ax.set_xlabel("$X$ [m]")
    ax.set_ylabel("$Y$ [m]")
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=8)
    figstyle.finish(ax)
    fig.tight_layout()
    p = os.path.join(FIG, "circle_trajectory.png")
    fig.savefig(p)
    plt.close(fig)
    print("saved", p)


def algo_compare():
    """Grouped bar chart: steady-state RMSE of four DRL algos vs. controller."""
    results = {
        "paper": {"line": 0.0109, "circle": 0.1424},
        "SAC":   {"line": 0.0156, "circle": 0.0953},
        "TD3":   {"line": 0.0236, "circle": 0.0984},
        "PPO":   {"line": 0.0756, "circle": 0.1187},
        "DDPG":  {"line": 0.1323, "circle": 0.1678},
    }
    names = list(results)
    x = np.arange(len(names))
    w = 0.36
    line = [results[n]["line"] for n in names]
    circ = [results[n]["circle"] for n in names]
    cols = [figstyle.COLORS[n] for n in names]

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    b1 = ax.bar(x - w/2, line, w, color=cols, edgecolor="#222222", lw=0.7,
                alpha=0.55)
    b2 = ax.bar(x + w/2, circ, w, color=cols, edgecolor="#222222", lw=0.7)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.002,
                    f"{b.get_height():.3f}", ha="center", va="bottom",
                    fontsize=7)
    # legend proxy for the line/circle shading
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="0.5", alpha=0.55, edgecolor="#222222",
                             label="line"),
                       Patch(facecolor="0.5", edgecolor="#222222",
                             label="circle")],
              loc="upper left")
    ax.set_xticks(x)
    ax.set_xticklabels(["controller" if n == "paper" else n for n in names])
    ax.set_ylabel("steady-state RMSE [m]")
    ax.set_ylim(0, 0.19)
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    figstyle.finish(ax)
    fig.tight_layout()
    p = os.path.join(FIG_RL, "algo_compare.png")
    fig.savefig(p)
    plt.close(fig)
    print("saved", p)


def rl_circle_traj():
    """SAC policy vs. model-based controller on the circle (uses trained model)."""
    try:
        from rl.evaluate import rollout_rl, PAPER_POSE0
    except Exception as e:                                   # pragma: no cover
        print("skip rl_circle_traj (rl deps unavailable):", e)
        return
    mp = os.path.join(ROOT, "rl", "models", "sac_mix")
    vn = mp + "_vecnorm.pkl"
    if not os.path.exists(mp + ".zip"):
        print("skip rl_circle_traj (sac_mix model missing)")
        return
    rl = rollout_rl(mp, vn, "circle", algo="sac")
    pp = simulate(traj_name="circle", T=60.0, dt=0.01,
                  pose0=PAPER_POSE0["circle"], use_dynamics=False, adaptive=True)
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    ax.plot(rl["xr"], rl["yr"], color=figstyle.COLORS["ref"], ls=(0, (5, 3)),
            lw=1.6, label="reference")
    ax.plot(pp["x"], pp["y"], color=figstyle.COLORS["TD3"], lw=1.6,
            label="controller")
    ax.plot(rl["x"], rl["y"], color=figstyle.COLORS["SAC"], lw=1.6,
            label="SAC policy")
    ax.plot(rl["x"][0], rl["y"][0], "o", color=figstyle.COLORS["DDPG"], ms=6,
            label="start")
    ax.set_xlabel("$X$ [m]")
    ax.set_ylabel("$Y$ [m]")
    ax.set_aspect("equal")
    ax.legend(loc="center", fontsize=7.5)
    figstyle.finish(ax)
    fig.tight_layout()
    p = os.path.join(FIG_RL, "circle_traj_sac_vs_paper.png")
    fig.savefig(p)
    plt.close(fig)
    print("saved", p)


if __name__ == "__main__":
    compare_adaptive()
    line_slip()
    circle_trajectory()
    algo_compare()
    rl_circle_traj()
