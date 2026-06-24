"""Simulation loop and plotting -- reproduces the paper's figures."""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .params import RobotParams, KinematicGains, DynamicGains
from .trajectories import make_trajectory
from .model import KinematicWMRPlant, DynamicWMRPlant, SlipProfile
from .controller import (
    KinematicAdaptiveController, DynamicBackstepping, pose_error,
)


def simulate(traj_name="line", T=60.0, dt=0.01, pose0=None, slip=None,
             use_dynamics=False, adaptive=True):
    """Run one closed-loop simulation.

    Parameters
    ----------
    traj_name : 'line' or 'circle'
    T, dt     : horizon and step [s]
    pose0     : initial robot pose [xG, yG, theta]; default = offset from ref.
    slip      : SlipProfile; default reproduces the t=30s/t=50s step changes.
    use_dynamics : if True, run the dynamic (torque) layer on the dynamic plant.
    adaptive  : if False, freeze the slip estimate at i=1 (no compensation) --
                useful as a baseline to show the effect of slip.
    """
    robot = RobotParams()
    traj = make_trajectory(traj_name)
    slip = slip or SlipProfile()

    # default initial pose: offset from the reference start to show convergence
    ref0, vr0, wr0 = traj.state(0.0)
    if pose0 is None:
        if traj_name == "line":
            pose0 = (1.6, 0.0, np.pi / 4)
        else:
            pose0 = (1.5, -0.4, 0.0)

    kin = KinematicAdaptiveController(robot, KinematicGains())
    if not adaptive:
        kin.adapt = lambda *a, **k: None  # freeze estimates at i=1

    if use_dynamics:
        plant = DynamicWMRPlant(robot, pose0=pose0)
        dyn = DynamicBackstepping(robot, DynamicGains())
    else:
        plant = KinematicWMRPlant(robot, pose0=pose0)
        dyn = None

    n = int(round(T / dt))
    log = {k: np.zeros(n) for k in
           ["t", "x", "y", "xr", "yr", "xe", "ye", "the",
            "sL_true", "sR_true", "sL_hat", "sR_hat", "v", "w"]}

    # low-pass filtered body angular velocity fed back to the kinematic layer
    # (enforces time-scale separation when the dynamic inner loop is active)
    w_filt = 0.0
    tau_f = 0.05  # filter time constant [s]

    for i in range(n):
        t = i * dt
        ref, vr, wr = traj.state(t)
        e = pose_error(plant.pose, ref)

        if use_dynamics:
            a = dt / (tau_f + dt)
            w_filt += a * (plant.w - w_filt)
            w_fb = w_filt
        else:
            w_fb = plant.w
        vc, wc = kin.virtual_velocity(e, vr, wr, w_fb)
        kin.adapt(e, vc, wc, dt)

        sL, sR = slip(t)
        wL, wR = kin.wheel_commands(vc, wc)   # slip-compensated wheel speeds
        if use_dynamics:
            # inner torque loop + plant run together at the fine substep so the
            # rate-damping derivative is meaningful (zero-order hold on wL, wR)
            h = dt / plant.n_sub
            for _ in range(plant.n_sub):
                tauL, tauR = dyn.torques(wL, wR, plant.wL, plant.wR,
                                         plant.v, plant.w, h)
                plant.micro_step(tauL, tauR, sL, sR, h)
        else:
            plant.step(wL, wR, sL, sR, dt)

        sL_hat, sR_hat = kin.slip_estimate
        log["t"][i] = t
        log["x"][i], log["y"][i] = plant.pose[0], plant.pose[1]
        log["xr"][i], log["yr"][i] = ref[0], ref[1]
        log["xe"][i], log["ye"][i], log["the"][i] = e
        log["sL_true"][i], log["sR_true"][i] = sL, sR
        log["sL_hat"][i], log["sR_hat"][i] = sL_hat, sR_hat
        log["v"][i], log["w"][i] = plant.v, plant.w

    return log


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_results(log, traj_name, outdir="figures", tag=""):
    os.makedirs(outdir, exist_ok=True)
    suffix = f"_{tag}" if tag else ""

    # 1. trajectory in the plane
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(log["xr"], log["yr"], "k--", lw=2, label="reference")
    ax.plot(log["x"], log["y"], "b-", lw=1.5, label="actual")
    ax.plot(log["x"][0], log["y"][0], "go", label="start")
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
    ax.set_title(f"WMR {traj_name} trajectory tracking")
    ax.axis("equal"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    f1 = os.path.join(outdir, f"{traj_name}_trajectory{suffix}.png")
    fig.savefig(f1, dpi=130); plt.close(fig)

    # 2. tracking errors
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(log["t"], log["xe"], label=r"$x_e$")
    ax.plot(log["t"], log["ye"], label=r"$y_e$")
    ax.plot(log["t"], log["the"], label=r"$\theta_e$")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("t [s]"); ax.set_ylabel("error")
    ax.set_title(f"WMR {traj_name} tracking error")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    f2 = os.path.join(outdir, f"{traj_name}_error{suffix}.png")
    fig.savefig(f2, dpi=130); plt.close(fig)

    # 3. slip-rate estimation
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(log["t"], log["sL_true"], "C0--", label=r"$s_L$ true")
    ax.plot(log["t"], log["sL_hat"], "C0-", label=r"$\hat{s}_L$")
    ax.plot(log["t"], log["sR_true"], "C3--", label=r"$s_R$ true")
    ax.plot(log["t"], log["sR_hat"], "C3-", label=r"$\hat{s}_R$")
    ax.set_xlabel("t [s]"); ax.set_ylabel("slip ratio")
    ax.set_title(f"WMR {traj_name} slip-rate estimation")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    f3 = os.path.join(outdir, f"{traj_name}_slip{suffix}.png")
    fig.savefig(f3, dpi=130); plt.close(fig)

    return [f1, f2, f3]
