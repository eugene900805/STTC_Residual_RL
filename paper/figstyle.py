"""Shared figure styling for the paper, so every figure looks consistent and
print-ready for an IEEE Transactions submission.

Design goals (learned from the reference article in example/):
  * a Times-like serif face for all text, matching the body font of the paper;
  * no in-figure titles -- the description belongs in the LaTeX/Word caption,
    not baked into the bitmap (avoids the redundant double caption);
  * light, unobtrusive grid; thin spines; restrained colours;
  * a single colour map reused for the four DRL algorithms everywhere.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# Times-like serif that ships with Liberation; fall back gracefully.
_SERIF = ["Liberation Serif", "Times New Roman", "DejaVu Serif", "serif"]
for _name in ("Liberation Serif", "DejaVu Serif"):
    try:
        font_manager.findfont(_name, fallback_to_default=False)
        _SERIF.insert(0, _name)
        break
    except Exception:
        continue

# One consistent palette for the four algorithms + the model-based controller.
COLORS = {
    "paper":  "#404040",   # model-based controller (neutral dark grey)
    "SAC":    "#1f77b4",   # blue
    "TD3":    "#2ca02c",   # green
    "PPO":    "#ff7f0e",   # orange
    "DDPG":   "#d62728",   # red
    "ref":    "#000000",   # reference trajectory
}
DPI = 220


def apply():
    """Install the shared rcParams. Call once before building any figure."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": _SERIF,
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#333333",
        "axes.grid": True,
        "grid.color": "#cccccc",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.6,
        "lines.linewidth": 1.8,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#bbbbbb",
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    })


def finish(ax):
    """Common per-axis cleanup: hide the top/right spines for a lighter look."""
    if isinstance(ax, (list, tuple)):
        for a in ax:
            finish(a)
        return
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3, width=0.7)
