"""Bar chart comparing all RL algorithms (SAC/PPO/TD3/DDPG) against the paper
controller on the steady-state RMSE for line and circle, in one figure.

    python -m rl.plot_algo_compare
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTDIR = "figures_rl"

# steady-state RMSE from rl.evaluate (slip steps at 30s/50s, paper pose0)
RESULTS = {
    "paper":         {"line": 0.0109, "circle": 0.1424},
    "SAC":           {"line": 0.0156, "circle": 0.0953},
    "PPO":           {"line": 0.0756, "circle": 0.1187},
    "TD3":           {"line": 0.0236, "circle": 0.0984},
    "DDPG":          {"line": 0.1323, "circle": 0.1678},
    "Residual\nRL (SAC)": {"line": 0.0180, "circle": 0.1068},
}


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    names = list(RESULTS)
    x = np.arange(len(names))
    w = 0.38
    line = [RESULTS[n]["line"] for n in names]
    circ = [RESULTS[n]["circle"] for n in names]

    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    colors = ["0.5", "C0", "C1", "C2", "C3", "C4"]
    b1 = ax.bar(x - w / 2, line, w, label="line", color=colors,
                edgecolor="k", alpha=0.55)
    b2 = ax.bar(x + w / 2, circ, w, label="circle", color=colors,
                edgecolor="k")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("steady-state RMSE")
    ax.set_title("RL algorithms vs. paper controller (slip steps at 30s/50s)\n"
                 "left bar = line (faded), right bar = circle")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(OUTDIR, "algo_compare.png")
    fig.savefig(path, dpi=140)
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
