# wmr_tb3 — Slipping trajectory tracking on TurtleBot3 (ROS 2)

Applies the paper's kinematic backstepping + adaptive slip controller
(Lu et al., ICIEA 2024) to TurtleBot3. Validated in Gazebo (ROS 2 Humble).

## Build

```bash
cd ~/STT/ros2_ws
colcon build --packages-select wmr_tb3
source install/setup.bash
```

## Run in Gazebo

```bash
export TURTLEBOT3_MODEL=burger
# 1) start the simulator
ros2 launch turtlebot3_gazebo empty_world.launch.py
# 2) (new terminal) start the tracking controller
source ~/STT/ros2_ws/install/setup.bash
ros2 run wmr_tb3 tracking_node --ros-args -p use_sim_time:=true \
  -p trajectory:=circle -p mode:=paper -p pose_source:=odom \
  -p circle_radius:=0.7 -p circle_wr:=0.18
```

`trajectory: circle|line`, `mode: paper|nocomp`, `pose_source: odom|amcl`.

## How it works

* Controller computes the slip-compensated **wheel speeds** (eq.11 with the
  adaptive `i^`), then inverts them back to a `cmd_vel` (v, w) through the
  robot's own `r, b`, so TB3's DYNAMIXEL velocity loop reproduces the inflated
  wheel speeds — i.e. the slip compensation is realised on hardware.
* Pose comes from `/odom` (or `/amcl_pose`). For real slip rejection use AMCL
  (LiDAR) so the pose is independent of wheel slip.
* The reference trajectory is rigidly transformed to start at the robot's start
  pose (position **and** heading) so tracking begins with ~zero error.

## Gazebo results (empty world, paper controller, odom pose)

| trajectory | steady-state RMSE |
|------------|-------------------|
| line  (v=0.12 m/s)        | ~0.038 m |
| circle (R=0.7 m, v=0.126) | ~0.0024 m |

TB3 params: r=0.033 m, b=0.08 m, d≈0; v ≤ 0.22 m/s.

## Slip experiment finding (sim-to-real gap)

Gazebo's Coulomb wheel friction gives only a narrow, uncontrollable slip window
on the light, slow TB3, and is a different model from the paper's proportional
slip.  So slip is injected at the wheel-command level (`slip_injector`, or
`run_experiment.py`) faithfully to the paper (v_wheel*(1-s), steps at 30/50 s).

Result on the real (dynamic) TB3 under injected slip (circle):

| controller | steady RMSE |
|------------|-------------|
| no-comp (pure backstepping feedback) | ~0.33 m (stable but degraded) |
| paper adaptive (eq.13/14) | DIVERGES (slip estimate hits clips) |

The paper's **kinematic** adaptive law assumes instantaneous velocity tracking
(q_dot = S(q) alpha).  The real robot's velocity dynamics (accel limit, lag)
violate this, so the parameter adaptation drifts/diverges -- robustly, across
rho = 10..2 and even with sigma-modification (leakage).

**Fix = velocity inner loop (the paper's dynamic layer).**  Adding a PI loop that
drives the *measured* body velocity (/odom twist) to the kinematic command
(vc, wc) -- its integral absorbs the persistent slip -- restores stable slip
compensation on the real robot:

| controller (circle under slip) | steady RMSE |
|--------------------------------|-------------|
| kinematic only (no-comp)       | ~0.34 m (drifts outward as slip grows) |
| kinematic + velocity inner loop | **~0.13 m** (stable, ~2.7x better) |
| kinematic adaptive (eq.13/14)  | diverges |

So on a real dynamic robot the slip compensation must close a velocity loop, not
rely on the kinematic adaptive feedforward alone.  (Without slip the plain
controller already tracks at 2.4 mm.)

## RL in the loop, on the real robot (Gazebo-trained SAC)

A SAC policy trained in an abstract Python sim **fails to transfer** to Gazebo
(spirals to the centre) -- a classic sim-to-real gap, not fixed by domain
randomisation over a simplified dynamics model.  Two things make it work:

1. **Physics-in-the-loop training** (`rl/gazebo_env.py`, `rl/train_gazebo.py`):
   every RL step advances the real Gazebo TB3, so training dynamics == deploy
   dynamics.
2. **Control-rate matching**: training and deployment must use the *same* fixed
   control rate.  Gazebo `pause/unpause` makes each step exactly `dt` of sim time
   (verified 0.102 s, deterministic), and both train and deploy run at
   **dt = 0.1 s (10 Hz, the TB3 rate)**.  With a mismatch (20 Hz train vs 5 Hz
   deploy) the policy gave 2.27 m; matched, it gives 0.099 m.

### Final 3-way result (circle under injected slip, 10 Hz)

| controller | steady RMSE |
|------------|-------------|
| kinematic only (no-comp)        | 0.361 m |
| kinematic + velocity inner loop | **0.027 m** (best, smoothest) |
| Gazebo-trained RL (SAC, 10 Hz)  | 0.099 m (works; noisier) |

Run (isolated, since the machine may be shared):
```bash
export ROS_DOMAIN_ID=42 GAZEBO_MASTER_URI=http://localhost:11346
# train:  python3 -m rl.train_gazebo --timesteps 60000 --tag tb3sac_gz10
# deploy: (unpause physics first) python3 run_experiment.py ; python3 plot_experiment.py
```

### Residual RL on top of the algorithm (algorithm + RL) -- BEST

`rl/gazebo_residual_env.py` + `rl/eval_residual.py`: base = the velocity-loop
controller, RL learns only a small residual (dv, dw) added on top, trained
physics-in-the-loop at 10 Hz.  This *combines* the model-based controller's
robustness with an RL correction:

| controller (circle under slip, 10 Hz) | steady RMSE |
|----------------------------------------|-------------|
| base velocity loop (algorithm)         | 0.0269 m |
| **base + RL residual (algorithm + RL)** | **0.0058 m** (4.7x better) |

The residual mainly kills the slip-step transients (t=30/50 s): the base spikes
to ~0.04-0.05 m and recovers slowly, while base+residual stays ~0.01 m.

**Takeaways:** on the real robot, model-based control (paper backstepping +
velocity inner loop) is robust and the best *single* method; a standalone
sim-trained RL works only with physics-in-the-loop training + matched control
rate (0.099 m); but **adding an RL residual on top of the model-based controller
("algorithm + RL") is the best of all (5.8 mm)** -- it inherits the controller's
stability and learns to reject the slip transients it leaves behind.

## Next steps (optional)

* AMCL localisation (`pose_source:=amcl`) with a map for slip-independent pose.
* Real TB3 hardware (LiDAR/mocap pose under slip).
