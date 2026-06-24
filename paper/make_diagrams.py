"""Generate the two schematic figures used in the paper:
  fig_model.png    - WMR with longitudinal wheel slip (kinematic model intro)
  fig_flow.png     - overall system / control architecture block diagram

Both are redrawn in a clean, serif, IEEE-print style (no overlapping labels,
no in-figure title) so they sit naturally beside the body text.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Arc

from paper import figstyle

OUT = os.path.dirname(__file__)
INK = "#222222"
BLUE = "#1f5fa8"
RED = "#c0392b"
GREY = "#8a8a8a"


# --------------------------------------------------------------------------- #
def model_diagram():
    """Differential-drive WMR body with the slip decomposition at one wheel."""
    figstyle.apply()
    fig, ax = plt.subplots(figsize=(5.4, 3.8))

    # --- world frame -------------------------------------------------------
    ax.annotate("", xy=(2.55, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", lw=1.1, color=INK))
    ax.annotate("", xy=(0, 2.25), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", lw=1.1, color=INK))
    ax.text(2.60, -0.02, r"$X$", fontsize=12, va="center")
    ax.text(-0.04, 2.32, r"$Y$", fontsize=12, ha="center")
    ax.text(-0.16, -0.16, r"$O$", fontsize=12)

    # --- robot body --------------------------------------------------------
    cx, cy, th = 1.35, 1.20, np.deg2rad(26)
    L, W = 0.92, 0.60
    cs, sn = np.cos(th), np.sin(th)
    R = np.array([[cs, -sn], [sn, cs]])
    corners = np.array([[-L/2, -W/2], [L/2, -W/2], [L/2, W/2], [-L/2, W/2]])
    pts = (corners @ R.T) + [cx, cy]
    ax.add_patch(plt.Polygon(pts, closed=True, fill=True, facecolor="#eef2f7",
                             edgecolor=INK, lw=1.5, zorder=2))

    # wheels (thick rounded segments on both flanks)
    wheel_centers = {}
    for sgn, name in ((1, "R"), (-1, "L")):
        wc = np.array([0, sgn*W/2]) @ R.T + [cx, cy]
        wv = np.array([cs, sn]) * 0.30
        ax.plot([wc[0]-wv[0], wc[0]+wv[0]], [wc[1]-wv[1], wc[1]+wv[1]],
                lw=5.5, color="#34495e", solid_capstyle="round", zorder=3)
        wheel_centers[name] = wc

    # centroid C
    ax.plot(cx, cy, "o", ms=4.5, color=INK, zorder=4)
    ax.annotate(r"$C\,(x,\,y)$", xy=(cx, cy), xytext=(cx-0.92, cy+0.30),
                fontsize=11, color=INK,
                arrowprops=dict(arrowstyle="-", lw=0.7, color=GREY))

    # wheel labels with short leader lines, placed in open space
    ax.annotate("right wheel", xy=wheel_centers["R"],
                xytext=(cx+1.02, cy-0.62), fontsize=9, color="#34495e",
                ha="left", va="center",
                arrowprops=dict(arrowstyle="-", lw=0.7, color=GREY))
    ax.annotate("left wheel", xy=wheel_centers["L"],
                xytext=(cx-1.18, cy-0.55), fontsize=9, color="#34495e",
                ha="left", va="center",
                arrowprops=dict(arrowstyle="-", lw=0.7, color=GREY))

    # --- heading / body velocity ------------------------------------------
    hv = np.array([cs, sn])
    tip = np.array([cx, cy]) + hv * 1.00
    ax.annotate("", xy=tuple(tip), xytext=(cx, cy),
                arrowprops=dict(arrowstyle="-|>", lw=2.0, color=BLUE))
    ax.text(tip[0]+0.02, tip[1]+0.10, r"$v$", color=BLUE, fontsize=13)

    # heading angle theta between +X-parallel dashed line and v
    ax.plot([cx, cx+0.95], [cy, cy], ls=(0, (5, 3)), lw=0.9, color=GREY)
    ax.add_patch(Arc((cx, cy), 1.15, 1.15, angle=0, theta1=0,
                     theta2=np.rad2deg(th), color=INK, lw=1.1))
    ax.text(cx+0.66, cy+0.13, r"$\theta$", fontsize=12, color=INK)

    # --- slip decomposition along the heading -----------------------------
    # realised body speed v (blue, above) vs commanded wheel travel; the red
    # dashed extension past the v tip is the slip deficit.
    real_tip = tip                                   # blue v arrow tip
    cmd_tip = np.array([cx, cy]) + hv * 1.46         # commanded would reach here
    ax.annotate("", xy=tuple(cmd_tip), xytext=tuple(real_tip),
                arrowprops=dict(arrowstyle="-|>", lw=1.5, color=RED,
                                ls=(0, (4, 2))))
    # small bracket marking the deficit between realised and commanded tips
    midd = (real_tip + cmd_tip) / 2
    ax.text(midd[0]-0.02, midd[1]+0.12, "slip", color=RED, fontsize=8.5,
            style="italic", ha="center")

    ax.text(2.02, 2.42, r"commanded wheel travel  $r\dot\alpha$",
            color=RED, fontsize=8.6, ha="left", alpha=0.9)
    ax.text(2.02, 2.22, r"realised body speed  $v = r\dot\alpha\,(1-s)$",
            color=BLUE, fontsize=8.6, ha="left")

    ax.set_xlim(-0.4, 3.5)
    ax.set_ylim(-0.4, 2.6)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    p = os.path.join(OUT, "fig_model.png")
    fig.savefig(p)
    plt.close(fig)
    print("saved", p)


# --------------------------------------------------------------------------- #
def _box(ax, xy, w, h, text, fc="white", ec=INK, fontsize=8.6, lw=1.3):
    x, y = xy
    box = FancyBboxPatch((x-w/2, y-h/2), w, h,
                         boxstyle="round,pad=0.012,rounding_size=0.05",
                         fc=fc, ec=ec, lw=lw, zorder=2)
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color=INK, zorder=3)
    return dict(x=x, y=y, w=w, h=h)


def _arrow(ax, p, q, color=INK, ls="-", lw=1.3):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12,
                                 lw=lw, color=color, ls=ls,
                                 shrinkA=1, shrinkB=1, zorder=1))


def flow_diagram():
    """Forward control chain + adaptive feedback + RL alternative path."""
    figstyle.apply()
    fig, ax = plt.subplots(figsize=(7.2, 3.5))

    y0 = 2.25          # main forward row
    y1 = 0.80          # feedback / plant row
    yr = 3.40          # RL row

    ref = _box(ax, (0.78, y0), 1.10, 0.66,
               "Reference\ntrajectory\n$(x_r,y_r,\\theta_r)$", fc="#eaf0fa")
    err = _box(ax, (2.20, y0), 1.02, 0.66,
               "Pose-error\ntransform\n(robot frame)", fc="#ffffff")
    kin = _box(ax, (3.72, y0), 1.18, 0.72,
               "Kinematic\nbackstepping\n$(v_c,\\omega_c)$ — (4)", fc="#e7f5ea")
    whl = _box(ax, (5.40, y0), 1.10, 0.72,
               "Wheel-speed\nmap — (6)\n$\\hat\\iota$ compensation", fc="#e7f5ea")
    dyn = _box(ax, (6.92, y0), 1.02, 0.66,
               "Dynamic /\nvelocity\ninner loop", fc="#ffffff")

    adp = _box(ax, (3.72, y1), 1.18, 0.66,
               "Slip adaptive\nestimator $\\hat\\iota$ — (5)", fc="#fcefe7")
    plant = _box(ax, (6.92, y1), 1.02, 0.66,
                 "WMR plant\n(+ true slip $s$)", fc="#f0e9fb")

    # forward chain
    for a, b in ((ref, err), (err, kin), (kin, whl), (whl, dyn)):
        _arrow(ax, (a["x"]+a["w"]/2, y0), (b["x"]-b["w"]/2, y0))

    # dynamic loop -> plant
    _arrow(ax, (dyn["x"], dyn["y"]-dyn["h"]/2), (plant["x"], plant["y"]+plant["h"]/2))
    ax.text(dyn["x"]+0.10, (y0+y1)/2, r"$\tau$ / cmd_vel", fontsize=8,
            color=INK, ha="left", va="center")

    # kinematic -> adaptive (uses pose error)
    _arrow(ax, (kin["x"]-0.28, kin["y"]-kin["h"]/2),
           (adp["x"]-0.28, adp["y"]+adp["h"]/2))
    ax.text(kin["x"]-0.52, (y0+y1)/2, r"$x_e$", fontsize=9, color=INK,
            ha="right", va="center")

    # adaptive estimate -> wheel-speed map
    _arrow(ax, (adp["x"]+adp["w"]/2, adp["y"]+0.10),
           (whl["x"]-0.20, whl["y"]-whl["h"]/2), color=RED)
    ax.text(5.02, 1.50, r"$\hat\iota$", fontsize=10, color=RED, ha="center")

    # feedback: plant -> pose-error transform (slip hidden); one cornered line
    fb_y = 0.18
    ax.add_patch(FancyArrowPatch((plant["x"], plant["y"]-plant["h"]/2),
                                 (plant["x"], fb_y), arrowstyle="-", lw=1.3,
                                 color=GREY, ls=(0, (4, 2))))
    ax.add_patch(FancyArrowPatch((plant["x"], fb_y), (err["x"], fb_y),
                                 arrowstyle="-", lw=1.3, color=GREY,
                                 ls=(0, (4, 2))))
    _arrow(ax, (err["x"], fb_y), (err["x"], err["y"]-err["h"]/2),
           color=GREY, ls=(0, (4, 2)))
    ax.text((err["x"]+plant["x"])/2, fb_y-0.16,
            "measured pose / velocity  (slip hidden)", ha="center",
            fontsize=7.8, color="#666666")

    # RL alternative path
    rl = _box(ax, (4.30, yr), 4.1, 0.52,
              "RL policy  (SAC / PPO / TD3 / DDPG)\nreplaces or residual-augments the controller",
              fc="#fff5dc", fontsize=8.2)
    _arrow(ax, (kin["x"], yr-0.26), (kin["x"], kin["y"]+kin["h"]/2),
           color="#e08a14")

    ax.set_xlim(0.0, 7.7)
    ax.set_ylim(-0.05, 3.8)
    ax.axis("off")
    ax.set_aspect("equal")
    fig.tight_layout()
    p = os.path.join(OUT, "fig_flow.png")
    fig.savefig(p)
    plt.close(fig)
    print("saved", p)


if __name__ == "__main__":
    model_diagram()
    flow_diagram()
