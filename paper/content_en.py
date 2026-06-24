"""English content of the paper."""
import os

ROOT = os.path.dirname(os.path.dirname(__file__))
COL = 3.3  # single body-column figure width [in]


def F(*p):
    return os.path.join(ROOT, *p)


def build(p):
    # ---------------- masthead ---------------- #
    p.title("Slipping Trajectory Tracking Control of a Wheeled Mobile Robot: "
            "Dynamics-Model Backstepping, a Four-Algorithm Deep "
            "Reinforcement-Learning Study, and TurtleBot3 Deployment")
    p.authors("First A. Author, Graduate Student, National Cheng Kung "
              "University, Tainan, Taiwan")

    p.abstract(
        "Abstract—",
        "Wheeled mobile robots (WMRs) operating on wet, icy, or loose terrain "
        "suffer from longitudinal wheel slip, which injects an unmodeled "
        "velocity loss and causes the tracking error to accumulate. This paper "
        "first reproduces the layered (kinematic plus dynamic) backstepping "
        "controller of Lu et al. that compensates slip through an online "
        "adaptive slip-ratio estimator, and validates it independently in "
        "Python and MATLAB. We then formulate the same task as a Markov "
        "decision process and train four model-free deep reinforcement-learning "
        "(DRL) algorithms—SAC, PPO, TD3 and DDPG—under identical "
        "observation, action, network and training budget, so that only the "
        "algorithm differs. Finally we deploy both the model-based controller "
        "and the learned policies on a TurtleBot3 Burger in Gazebo through "
        "ROS 2. In simulation, off-policy SAC and TD3 match or exceed the "
        "adaptive controller, reducing the circular-path steady-state error "
        "from 0.142 m to about 0.095 m. On the robot, a model-based "
        "velocity inner loop is the most robust single method (0.027 m), "
        "while an “algorithm-plus-RL-residual” scheme attains the best "
        "result (0.0058 m). We also show that naive sim-trained policies "
        "fail to transfer unless training is performed in the physics loop and "
        "the control rate is matched between training and deployment.")

    p.index_terms(
        "Index Terms—",
        "Adaptive control, backstepping, deep reinforcement learning, ROS 2, "
        "sim-to-real transfer, trajectory tracking, wheel slip, wheeled mobile "
        "robot.")

    p.start_body()

    # ---------------- I. Introduction ---------------- #
    p.h1("I. Introduction")
    p.para(
        "Trajectory tracking is a core capability of autonomous wheeled mobile "
        "robots (WMRs). The classical differential-drive kinematic model "
        "assumes pure rolling without slipping, so the body velocity is a rigid "
        "function of the wheel angular velocities. In practice, however, robots "
        "frequently traverse wet floors, thin ice, sand or loose soil, on which "
        "the contact patch slips. Longitudinal slip means that only a fraction "
        "of the commanded wheel travel is converted into body motion; the "
        "resulting velocity deficit is unknown, time-varying and asymmetric "
        "between the left and right wheels, and it makes the tracking error "
        "drift without bound when no compensation is applied.")
    p.para(
        "Lu et al. [1] addressed this problem with a dynamics-model controller "
        "that augments a kinematic backstepping law with an online adaptive "
        "estimator of the slip ratio, and proved asymptotic convergence of the "
        "pose and velocity errors through a Lyapunov argument. Their results, "
        "obtained in MATLAB, show that the adaptive law learns the true slip "
        "and drives the straight-line error to zero. Two questions motivate the "
        "present work. First, how does such a model-based controller compare "
        "with modern model-free deep reinforcement learning (DRL), which "
        "observes only the tracking error and must reject the slip implicitly? "
        "Second, do either of these approaches survive deployment on a real "
        "robot stack with actuator dynamics, latency and a finite control rate?")
    p.para(
        "This paper makes the following contributions. (i) We reproduce the "
        "layered backstepping plus adaptive-slip controller from scratch and "
        "cross-validate it in two independent implementations (Python and "
        "MATLAB). (ii) We cast the slipping tracking task as a Markov decision "
        "process and conduct a controlled comparison of four DRL algorithms "
        "(SAC, PPO, TD3, DDPG) under identical observation, action, network and "
        "training budget. (iii) We deploy both families on a TurtleBot3 in "
        "Gazebo via ROS 2 and quantify the sim-to-real gap, showing that a "
        "model-based velocity inner loop is the most robust single method and "
        "that combining it with a small learned residual yields the lowest "
        "error overall. The overall software and control architecture is "
        "summarized in Fig. 2.")

    # ---------------- II. Related work ---------------- #
    p.h1("II. Related Work")
    p.para(
        "The problem studied here sits at the intersection of three lines of "
        "research: model-based control of slipping wheeled robots, "
        "learning-based trajectory tracking, and hybrid schemes that combine the "
        "two. We review each in turn and position our contribution accordingly.")
    p.h2("A. Model-Based Slip Compensation and Adaptive Control")
    p.para(
        "Trajectory tracking for nonholonomic wheeled mobile robots is "
        "classically addressed by backstepping the kinematics into the dynamics "
        "[2], under the assumption of pure rolling without slipping. Once slip is "
        "admitted, the velocity map between the wheels and the body is no longer "
        "exact, and the controller must either estimate or reject the deficit. "
        "Several remedies have been proposed: online adaptive estimation of the "
        "slip ratio embedded in the tracking law [1]; sliding-mode controllers "
        "that treat skidding and slipping as a bounded matched disturbance [9]; "
        "nonlinear disturbance observers that lump the unmodeled velocity loss "
        "into an estimated term and cancel it [10]; and model-predictive control "
        "that encodes slip and actuation limits as explicit constraints [11]. "
        "These methods are provably convergent through Lyapunov or Barbalat "
        "arguments, but they depend on an accurate dynamic model and on "
        "assumptions such as instantaneously realizable velocity and bounded, "
        "slowly varying slip. The controller we reproduce [1] belongs to this "
        "family; it is representative because it makes the slip explicit through "
        "an adaptive estimate, which gives us a transparent model-based baseline "
        "to compare against a model-free policy.")
    p.h2("B. Learning-Based Trajectory Tracking")
    p.para(
        "Deep reinforcement learning produces feedback controllers for "
        "continuous robotic tasks directly from interaction, without an explicit "
        "model of the disturbance. The standard continuous-control algorithms are "
        "the off-policy actor-critic methods DDPG [6], TD3 [5] and the "
        "entropy-regularized SAC [3], together with the on-policy PPO [4]; we "
        "benchmark all four under an identical budget so that only the update "
        "rule differs. Reinforcement learning has been demonstrated on mobile-"
        "robot and vehicle path tracking, on agile legged locomotion [12], and on "
        "end-to-end autonomous driving [13], with sample inefficiency and the "
        "sim-to-real gap as the recurring costs. Most reinforcement-learning "
        "work on wheeled-robot tracking either assumes no slip or trains in a "
        "purely kinematic simulator; by contrast, we expose the policy to an "
        "asymmetric, step-changing slip and deliberately hide the slip from the "
        "observation, so that the agent must reconstruct and reject the "
        "disturbance from the tracking error alone — the same information the "
        "adaptive law uses.")
    p.h2("C. Hybrid and Residual Control, and Sim-to-Real Transfer")
    p.para(
        "A pragmatic middle ground keeps a stabilizing model-based core and "
        "learns only a corrective residual on top of it [14], so that the policy "
        "inherits the stability of the base controller while absorbing effects "
        "that are hard to model. Transfer of such controllers from simulation to "
        "hardware is commonly approached by domain randomization [15] and, more "
        "fundamentally, by reducing the mismatch between the training and "
        "deployment dynamics. Our deployment study supports the latter view: for "
        "this task, training inside the physics loop and matching the control "
        "rate between training and deployment matter more than randomizing a "
        "simplified model. We evaluate a residual policy on top of a model-based "
        "velocity inner loop and show that it attains the lowest on-robot error, "
        "whereas a residual on a fully adaptive base is absorbed by the "
        "adaptation and cannot change the steady-state error.")

    # ---------------- III. WMR slipping model ---------------- #
    p.h1("III. WMR Slipping Model")
    p.figure(F("paper", "fig_model.png"),
             "Fig. 1.  Differential-drive WMR with centroid C, heading "
             "θ and longitudinal wheel slip. Only a fraction (1−s) of "
             "the commanded wheel travel is realized as body motion.")
    p.para(
        "Consider a differential-drive WMR with wheel radius r and half-track b, "
        "whose configuration is q = [x, y, θ]ᵀ, the position of the "
        "centroid C and the heading angle. For wheel k ∈ {L, R} with angular "
        "velocity α̇ₖ, the longitudinal slip ratio sₖ is "
        "defined as the relative velocity loss at the contact point,")
    p.equation(1)
    p.para(
        "so that the slip-free case sₖ = 0 recovers pure rolling and "
        "sₖ → 1 denotes full spinning. Writing the slipped body linear "
        "and angular velocities as v and ω, the kinematics on the plane are")
    p.equation(2)
    p.para(
        "The key modeling insight is that v and ω are produced by the "
        "slipped wheel speeds rα̇ₖ(1−sₖ) rather than by "
        "the commanded speeds. If the controller knew the slip it could "
        "pre-compensate the wheel commands; the difficulty is that sₖ is "
        "unmeasured. This motivates either an online estimator (Section IV) or "
        "a learning policy that rejects the slip from error feedback "
        "(Section V).")
    p.para(
        "Two assumptions underlie the model. First, the robot is subject to the "
        "nonholonomic rolling constraint, so lateral skidding is neglected and "
        "the disturbance enters only through the longitudinal channel; this is "
        "consistent with the wet- or icy-floor scenarios that motivate the "
        "work, where the wheels spin rather than slide sideways. Second, the "
        "slip ratio is bounded away from unity, so the inverse-slip factor "
        "1/(1−s) remains finite and the wheel-speed map is well posed. Both "
        "assumptions are standard and hold throughout our experiments, in which "
        "the per-wheel slip never exceeds 0.25.")

    # ---------------- III. Controller design ---------------- #
    p.h1("IV. Model-Based Controller Design")
    p.wide_figure(F("paper", "fig_flow.png"),
             "Fig. 2.  System architecture. The reference is transformed into a "
             "robot-frame pose error; kinematic backstepping produces a virtual "
             "velocity; the adaptive estimator and the wheel-speed map inject "
             "slip compensation; an optional dynamic / velocity inner loop "
             "realizes the command. A DRL policy can replace or "
             "residual-augment the controller.")
    p.h2("A. Pose Error and Kinematic Backstepping")
    p.para(
        "Following the standard tracking formulation, the world-frame pose "
        "error between the reference qᵣ and the robot q is rotated into the "
        "robot frame,")
    p.equation(3)
    p.para(
        "where xₑ, yₑ and θₑ are the longitudinal, lateral "
        "and heading errors. A backstepping virtual control law drives this "
        "error to zero,")
    p.equation(4)
    p.para(
        "with positive gains kₓ, kᵧ and kθ. On the slip-free "
        "model, substituting (4) into the error dynamics yields a Lyapunov "
        "function whose derivative is negative semidefinite, so (xₑ, "
        "yₑ, θₑ) → 0.")
    p.para(
        "The same Lyapunov function extends to the slipping case once the "
        "adaptive estimate is included as an additional state: the candidate is "
        "augmented with a quadratic term in the estimation error, and the "
        "adaptation law (5) is chosen so that its time derivative cancels the "
        "indefinite cross term, leaving a negative semidefinite total. "
        "Boundedness of all signals then follows, and Barbalat's lemma gives "
        "convergence of the pose error; the estimate itself need not converge "
        "to the true slip unless the reference is persistently exciting, which "
        "explains the small circular-path residual reported below.")
    p.h2("B. Adaptive Slip Compensation")
    p.para(
        "Because the true slip is unknown, the controller estimates the "
        "inverse-slip factor îₖ = 1/(1−sₖ) online with the "
        "adaptation law")
    p.equation(5)
    p.para(
        "where γₖ > 0 is the adaptation gain. The estimate is then "
        "injected into the wheel-speed command so that, after the plant applies "
        "the true slip, the realized body velocity equals the virtual command,")
    p.equation(6)
    p.para(
        "When îₖ → 1/(1−sₖ) the compensation is exact "
        "and tracking is maintained despite the slip. In our reproduction the "
        "adaptive law recovers the true slip on the straight line and drives "
        "the error to zero; on the circle the estimator settles at a slightly "
        "biased equilibrium, leaving an approximately 0.1 m radial "
        "residual. This residual is consistent with the convergence analysis: "
        "a constant-curvature reference is only weakly persistently exciting, "
        "so the adaptation converges to a biased fixed point rather than to the "
        "true slip, while the pose error remains bounded but nonzero.")
    p.h2("C. Dynamic Layer")
    p.para(
        "The kinematic law assumes that the commanded body velocity is achieved "
        "instantaneously. A real platform has bounded wheel acceleration and "
        "actuator latency, so the dynamic layer closes a torque (or velocity) "
        "loop on top of the kinematic command,")
    p.equation(7)
    p.para(
        "with inertia matrix M, Coriolis term C and wheel inertia I_w. On the "
        "robot we realize this layer as a proportional-integral velocity inner "
        "loop on cmd_vel, whose integral action absorbs the steady slip; this "
        "turns out to be essential for stability on hardware, as discussed in "
        "Section VII.")

    # ---------------- IV. RL formulation ---------------- #
    p.h1("V. Reinforcement-Learning Formulation")
    p.para(
        "We pose slipping trajectory tracking as a Markov decision process so "
        "that a model-free agent can be compared head-to-head with the "
        "model-based controller on the same plant and slip profile. The "
        "observation deliberately hides the slip,")
    p.para(
        "o = [xₑ, yₑ, sinθₑ, cosθₑ, vᵣ, "
        "ωᵣ], forcing the policy to reject the disturbance from error "
        "feedback exactly as the adaptive law must. The action is the body "
        "velocity command a = [v, ω], mapped to wheel speeds without slip "
        "compensation; the plant then applies the true, time-varying slip. The "
        "reward penalizes the quadratic tracking error and a small control "
        "effort,")
    p.equation(8)
    p.para(
        "with λ a small weight. Each episode randomizes the initial pose "
        "offset, the slip magnitudes and the step-change instants over a mix of "
        "straight-line and circular references. To make the comparison fair, "
        "all four algorithms share the same observation, action, two-layer "
        "256×256 multilayer-perceptron policy, learning rate 3×10⁻"
        "⁴, discount 0.99 and a 300 000-step budget; only the learning "
        "algorithm changes. We study SAC (entropy-regularized off-policy), TD3 "
        "(twin-critic deterministic off-policy), DDPG (deterministic "
        "off-policy) and PPO (clipped on-policy). We additionally consider a "
        "residual scheme in which a model-based controller provides the base "
        "command and the policy learns only a small additive correction.")
    p.para(
        "Table I lists the hyperparameters shared by all four agents. Holding "
        "them fixed isolates the effect of the learning algorithm itself: any "
        "difference in the results below is attributable to the update rule "
        "(on-policy vs. off-policy, single vs. twin critic, deterministic vs. "
        "stochastic actor) rather than to tuning. We use the Stable-Baselines3 "
        "implementations with their default algorithm-specific settings, and "
        "train on a single CPU process, which is sufficient because the "
        "bottleneck is the environment step rather than the small policy "
        "network.")
    p.table("TABLE I",
            "Shared Hyperparameters of the Four DRL Agents",
            ["Setting", "Value"],
            [["Policy network", "MLP, 256 × 256, ReLU"],
             ["Observation / action dim.", "6 / 2"],
             ["Learning rate", "3 × 10⁻⁴"],
             ["Discount γ", "0.99"],
             ["Training budget", "300 000 steps"],
             ["Replay buffer (off-policy)", "3 × 10⁵"],
             ["Rollout length (PPO)", "2048"],
             ["Exploration noise (TD3/DDPG)", "N(0, 0.1)"]])

    p.para(
        "The four algorithms differ chiefly in how they explore and how they "
        "estimate value. DDPG learns a single deterministic actor and critic "
        "and is known to be brittle, over-estimating the value and diverging "
        "when the noise is mistuned. TD3 adds twin critics, delayed policy "
        "updates and target smoothing, which directly address that brittleness. "
        "SAC further maximizes an entropy bonus, yielding a stochastic actor "
        "that explores more thoroughly and is the least sensitive to "
        "hyperparameters. PPO, the only on-policy method here, is robust and "
        "simple but less sample-efficient because it discards each batch after a "
        "few epochs. These properties predict the ranking we observe, and "
        "underline that for a slip-rejection task the exploration mechanism is "
        "as important as the network or the reward.")

    # ---------------- V. Simulation results ---------------- #
    p.h1("VI. Simulation Results")
    p.para(
        "We evaluate on the two references used throughout the literature, a "
        "straight line and a circle, because they probe complementary "
        "behaviours: the line exercises longitudinal regulation with a fixed "
        "heading, whereas the circle couples linear and angular motion and "
        "therefore exposes any steady-state bias in the slip compensation. All "
        "controllers and policies are run from the same initial offset and the "
        "same slip schedule, and every number is the mean over the final fifth "
        "of the run so that start-up transients do not contaminate the metric.")
    p.para(
        "The robustness scenario follows the original paper: the right-wheel "
        "slip steps up at t = 30 s and the left-wheel slip steps at "
        "t = 50 s. Steady-state error is the root-mean-square (RMS) of the "
        "pose error over the final 20 % of the run.")
    p.wide_figure(F("figures", "compare_adaptive.png"),
             "Fig. 3.  Tracking-error norm with and without adaptive slip "
             "compensation, for the line (left) and circle (right). The slip "
             "steps at t = 30 s and 50 s are rejected by the adaptive "
             "law.")
    p.figure(F("figures", "line_slip.png"),
             "Fig. 4.  Online slip-ratio estimation on the straight line. The "
             "adaptive estimates ŝ_L and ŝ_R track the true left/right "
             "slip and re-converge within a few seconds of each step change "
             "at t = 30 s and 50 s.")
    p.para(
        "Table II confirms the reproduction. The adaptive controller reduces the "
        "straight-line error roughly sixteenfold relative to the uncompensated "
        "baseline and remains clearly better on the circle. The MATLAB port "
        "matched the Python results to four significant figures, giving us "
        "confidence in the implementation.")
    p.table("TABLE II",
            "Steady-State RMSE: Model-Based Controller (Simulation)",
            ["Scenario", "Line [m]", "Circle [m]"],
            [["Kinematic + adaptive (paper)", "0.011", "0.142"],
             ["Dynamic cascade + adaptive", "0.018", "0.143"],
             ["No compensation (baseline)", "0.170", "0.223"]])
    p.para(
        "Table III reports the four-algorithm DRL comparison against the "
        "model-based controller, and Fig. 5 visualizes it. Two off-policy "
        "algorithms, SAC and TD3, are the strongest: both converge to about "
        "2 cm on the line and, notably, both beat the adaptive controller "
        "on the circle, where the model-based estimator leaves a residual. PPO "
        "is looser on the line but still beats the controller on the circle, "
        "whereas DDPG—lacking the entropy exploration of SAC and the "
        "twin-critic stabilization of TD3—is the most fragile and the "
        "worst overall. The ranking SAC ≈ TD3 ≫ PPO ≫ DDPG is "
        "consistent with the general reputation of these methods on continuous "
        "control.")
    p.table("TABLE III",
            "Steady-State RMSE: Four DRL Algorithms vs. Controller",
            ["Method", "Line [m]", "Circle [m]"],
            [["Model-based controller", "0.0109", "0.1424"],
             ["SAC", "0.0156", "0.0953"],
             ["TD3", "0.0236", "0.0984"],
             ["PPO", "0.0756", "0.1187"],
             ["DDPG", "0.1323", "0.1678"]])
    p.figure(F("figures_rl", "algo_compare.png"),
             "Fig. 5.  Steady-state RMSE of the four DRL algorithms and the "
             "model-based controller on the line and circle references.")
    p.figure(F("figures_rl", "circle_traj_sac_vs_paper.png"),
             "Fig. 6.  Circular-path tracking: the SAC policy (no slip "
             "observation) versus the model-based controller and the "
             "reference.")
    p.para(
        "The take-away from simulation is that a purely model-free policy that "
        "never observes the slip can match, and on the circle exceed, a "
        "model-based adaptive controller—provided the algorithm is chosen "
        "well. The choice of DRL algorithm matters as much as the decision to "
        "use DRL at all.")
    p.para(
        "It is worth emphasizing what the policy is not given. The slip ratio, "
        "its step times and the reference curvature are all hidden; the agent "
        "sees only the instantaneous pose error and the reference velocity. "
        "That the best agents nonetheless reject an asymmetric, step-changing "
        "slip demonstrates that the disturbance is observable from the error "
        "signal alone, which is the same property the adaptive law exploits. "
        "The practical difference is that the controller encodes this knowledge "
        "analytically and converges immediately, whereas the policy must "
        "discover it over 300 000 steps but is then free of the modeling "
        "bias that leaves the controller a residual on the circle.")
    p.para(
        "Figure 4 makes the mechanism concrete for the model-based controller: "
        "the slip estimates sit on top of the true values during each constant "
        "phase and recover within a few seconds of each step, which is exactly "
        "the transient visible as the small bumps at t = 30 s and 50 s in "
        "Fig. 3. The learned policies show a qualitatively similar recovery "
        "without ever estimating the slip explicitly, confirming that the "
        "step disturbance is reconstructed implicitly inside the network.")

    p.para(
        "The per-algorithm trajectories in Figs. 5 and 6 reinforce the table. "
        "On the circle the SAC path hugs the reference more tightly than the "
        "model-based controller, whose orbit is offset outward by the residual; "
        "on the line all competent methods overlay the reference once the "
        "initial error has decayed. The weaker agents instead show a visibly "
        "larger and slower-decaying error, which is why a single scalar RMSE, "
        "while convenient, understates how differently the algorithms behave "
        "during the transient that follows each slip step.")

    # ---------------- VI. Deployment ---------------- #
    p.h1("VII. TurtleBot3 Deployment in Gazebo")
    p.para(
        "We next deploy the controllers on a TurtleBot3 Burger in Gazebo using "
        "ROS 2 Humble. The controller output is converted to cmd_vel and "
        "realized by the DYNAMIXEL velocity loop; pose is taken from odometry. "
        "Because Gazebo Coulomb friction produces a narrow and poorly "
        "controllable slip on a light, slow robot, we inject the paper's "
        "proportional slip at the wheel-command level so that the disturbance "
        "is reproducible and faithful to the model.")
    p.para(
        "The controller is a ROS 2 node that subscribes to odometry, rigidly "
        "transforms the reference so it starts at the current robot pose, and "
        "publishes cmd_vel at a fixed rate. The slip-compensated wheel speeds "
        "of (6) are converted back to a body twist through the TurtleBot3 wheel "
        "radius and track so that the DYNAMIXEL velocity loop realizes the "
        "compensation, and the inverse-slip estimate is clipped to a safe range "
        "to prevent wind-up. The velocity inner loop is a small proportional-"
        "integral regulator on the measured twist with deliberately gentle "
        "gains, since aggressive gains excite an oscillation against the "
        "actuator dynamics.")
    p.figure(F("figures_tb3", "tb3_gazebo_tracking.png"),
             "Fig. 7.  TurtleBot3 tracking in the Gazebo simulator: circular "
             "(top) and straight-line (bottom) references with the resulting "
             "error transients.", width=COL)
    p.para(
        "Two findings dominate the hardware-level experiments. First, the "
        "kinematic adaptive law, which is stable in the abstract simulation, "
        "diverges on the dynamic robot: it assumes the commanded velocity is "
        "reached instantly, but real velocity lags, so the estimator winds up "
        "against its limits. Replacing the adaptive law with a model-based "
        "velocity inner loop—the dynamic layer realized as a PI loop on "
        "cmd_vel—restores stability, because the integral term absorbs the "
        "steady slip. Second, a policy trained only in the abstract simulator "
        "does not transfer at all: it spirals into the center of the circle, a "
        "textbook sim-to-real gap. Domain randomization over a simplified "
        "dynamics model does not close it either.")
    p.para(
        "The gap closes only when two conditions are met. (i) The policy is "
        "trained inside the Gazebo physics loop, so the training dynamics equal "
        "the deployment dynamics. (ii) The control rate is matched between "
        "training and deployment; we enforce a deterministic time step by "
        "pausing and unpausing the simulator, and run both at 10 Hz. With a "
        "mismatched rate (20 Hz training, 5 Hz deployment) the policy "
        "produces 2.27 m of error; once the rate is matched it produces "
        "0.099 m. Table IV and Figs. 7–9 summarize the on-robot "
        "comparison under injected slip.")
    p.para(
        "The deterministic time step deserves emphasis because it is the single "
        "change that turns a failing policy into a working one. We pause the "
        "simulator, publish the action, unpause for exactly one control period "
        "of simulated time, and pause again before the network is queried, so "
        "that the elapsed simulated time per step is constant regardless of how "
        "long the computation takes or how fast the simulator runs. Training "
        "and deployment then see identical step semantics, and the policy is no "
        "longer sensitive to wall-clock jitter, which is the hidden variable "
        "that breaks naive deployment.")
    p.table("TABLE IV",
            "Steady-State RMSE on TurtleBot3 (Gazebo, Circle, With Slip, 10 Hz)",
            ["Controller", "RMSE [m]"],
            [["Pure kinematic", "0.361"],
             ["Gazebo-trained RL (SAC)", "0.099"],
             ["Velocity inner loop (model-based)", "0.027"],
             ["Velocity loop + RL residual", "0.0058"]])
    p.figure(F("figures_tb3", "residual_vs_base.png"),
             "Fig. 8.  Algorithm-plus-RL: a small learned residual on top of "
             "the velocity inner loop mainly suppresses the slip-step "
             "transients, improving the steady error 4.7-fold.", width=COL)
    p.para(
        "The model-based velocity inner loop is the most robust single method "
        "on the robot (0.027 m) and the smoothest. The Gazebo-trained "
        "policy works once the rate is matched but is noisier (0.099 m). "
        "The best result is obtained by an algorithm-plus-RL scheme: with the "
        "velocity loop as the base and a learned residual on top, the error "
        "drops to 0.0058 m, because the residual chiefly cancels the "
        "transients caused by the slip steps while inheriting the stability of "
        "the model-based base.")

    p.wide_figure(F("figures_tb3", "slip_all.png"),
             "Fig. 9.  TurtleBot3 under injected slip in Gazebo: trajectory "
             "(left) and tracking error (right) for the pure kinematic "
             "controller, the model-based velocity inner loop, and the "
             "Gazebo-trained SAC policy.")

    p.para(
        "A practical caveat concerns localization. Throughout these experiments "
        "the pose is read from wheel odometry, which is itself corrupted by the "
        "very slip we study, so the reported on-robot errors are optimistic "
        "relative to a ground-truth measurement. A slip-immune estimator such "
        "as LiDAR scan matching or AMCL would both tighten the numbers and, "
        "more importantly, prevent the controller from chasing a biased pose. "
        "We regard the odometry results as a controlled relative comparison "
        "between methods rather than an absolute accuracy claim.")

    # ---------------- VII. Discussion ---------------- #
    p.h1("VIII. Discussion")
    p.para(
        "Three lessons emerge. First, in clean simulation a model-free policy "
        "can outperform a model-based adaptive controller, but only for the "
        "better off-policy algorithms; the spread between SAC and DDPG is "
        "larger than the gap between the controller and SAC. Second, on real "
        "hardware the model-based velocity loop is the robust workhorse, "
        "because its integral action handles the slip without relying on a "
        "fragile kinematic assumption. Third, the most accurate result couples "
        "the two: the controller supplies stability and the learned residual "
        "supplies the fine, transient correction the controller does not "
        "produce. A residual on a fully adaptive base is, by contrast, "
        "absorbed by the adaptation and cannot change the steady error—a "
        "structural limitation we verified with a constant-residual test.")

    p.para(
        "Several limitations remain. Pose is estimated from wheel odometry, "
        "which itself drifts under slip; a slip-immune source such as LiDAR or "
        "AMCL would tighten the on-robot numbers. The slip is injected at the "
        "wheel-command level rather than emerging from contact physics, a "
        "deliberate choice for reproducibility that nonetheless abstracts away "
        "tyre behaviour. Finally, all robot experiments are in Gazebo; physical "
        "validation is left to future work, although matching the control rate "
        "and training in the physics loop are precisely the steps that most "
        "ease a further transfer to hardware.")
    p.para(
        "More broadly, the results suggest a pragmatic division of labour. A "
        "model-based core should own stability and the bulk of the "
        "disturbance rejection, because its behaviour is predictable and its "
        "guarantees survive deployment. A learned residual should own the "
        "fine, hard-to-model corrections—here, the brief transients that "
        "follow each slip step—where an analytic law is either unavailable or "
        "too conservative. This pairing is what gives the lowest error on the "
        "robot and is the most defensible way to bring learning into a "
        "safety-relevant control loop.")

    p.para(
        "Finally, the study is reproducible end to end: the same Python package "
        "defines the plant for the controller, for the abstract DRL training and "
        "for the residual scheme, and the ROS 2 node reuses the identical "
        "control law on the robot. This shared implementation is what lets us "
        "attribute differences to the algorithm rather than to incidental "
        "modeling choices, and it is, in our view, a precondition for a fair "
        "comparison between classical and learned control.")

    # ---------------- VIII. Conclusion ---------------- #
    p.h1("IX. Conclusion")
    p.para(
        "We reproduced a dynamics-model slipping trajectory-tracking controller "
        "and validated it in Python and MATLAB; benchmarked four DRL algorithms "
        "under identical conditions, finding SAC and TD3 competitive with or "
        "better than the controller in simulation; and deployed both families "
        "on a TurtleBot3 in Gazebo. On the robot, a model-based velocity inner "
        "loop is the most robust single method, and an algorithm-plus-RL "
        "residual achieves the lowest error. Future work includes "
        "LiDAR/AMCL localization that is immune to odometry drift under slip, "
        "validation on physical hardware, and reducing the residual policy's "
        "control noise.")

    # ---------------- Appendix ---------------- #
    p.h1("Appendix")
    p.h2("A. Training Procedure")
    p.para(
        "Reward shaping was kept deliberately minimal: a quadratic pose-error "
        "term with a small control penalty and no curriculum, hand-crafted "
        "potential, or slip-specific term. This avoids encoding prior knowledge "
        "of the disturbance into the reward and keeps the comparison between "
        "the four algorithms about exploration and value estimation rather than "
        "about who received the more informative signal.")
    p.para(
        "Algorithm 1 summarizes the common training loop. The episode randomly "
        "selects a straight-line or circular reference, samples an initial pose "
        "offset and a piecewise-constant slip profile with one step change per "
        "wheel, and rolls out the chosen agent on the slipping plant. The same "
        "loop is used for all four algorithms; only the policy-update operator "
        "differs.")
    p.algorithm_box("Algorithm 1  Slip-rejection policy training", [
        "Require: agent A in {SAC, PPO, TD3, DDPG}, budget N",
        " 1: initialize policy pi, value/critic, (replay buffer B)",
        " 2: for step = 1 .. N do",
        " 3:    if episode end then",
        " 4:       traj <- rand({line, circle});  q0 <- q_ref + offset",
        " 5:       s_L, s_R <- piecewise slip with one step each",
        " 6:    o <- [x_e, y_e, sin th_e, cos th_e, v_r, w_r]",
        " 7:    a <- pi(o) + exploration;  (v,w) <- scale(a)",
        " 8:    map (v,w) -> wheel speeds (no slip comp.)",
        " 9:    apply true slip; integrate plant; observe o'",
        "10:    r <- -(x_e^2 + y_e^2 + th_e^2) - lambda||a||^2",
        "11:    store / accumulate (o,a,r,o'); update A",
        "12: return pi",
    ])
    p.h2("B. Model and Simulation Parameters")
    p.para(
        "Table V lists the principal robot and simulation parameters. The "
        "abstract simulation uses the paper's values; the TurtleBot3 experiments "
        "use the Burger geometry, whose top speed of 0.22 m/s forces the "
        "trajectories to be scaled down by roughly an order of magnitude.")
    p.table("TABLE V", "Principal Model and Simulation Parameters",
            ["Parameter", "Abstract sim.", "TurtleBot3"],
            [["Wheel radius r [m]", "0.05", "0.033"],
             ["Half-track b [m]", "0.10", "0.080"],
             ["Max. linear speed [m/s]", "2.0", "0.22"],
             ["Control rate [Hz]", "20", "10"],
             ["Slip step times [s]", "30 / 50", "30 / 50"],
             ["Episode horizon [s]", "60", "60"]])
    p.h2("C. Additional Results")
    p.para(
        "The supplementary figures below are provided for completeness. They "
        "show the abstract-simulation circle that underlies Table II and the "
        "effect of the velocity inner loop that underlies Table IV, and they "
        "use the same slip schedule and metric as the main results so that the "
        "two layers of experiment can be read against one another directly.")
    p.figure(F("figures", "circle_trajectory.png"),
             "Fig. 10.  Circular-path tracking of the model-based controller in "
             "the abstract simulation, with the slip steps applied at "
             "t = 30 s and 50 s.")
    p.figure(F("figures_tb3", "slip_veloop.png"),
             "Fig. 11.  Effect of the model-based velocity inner loop on the "
             "TurtleBot3 under slip: the integral action absorbs the steady "
             "velocity deficit that the pure kinematic law cannot.")

    # ---------------- References ---------------- #
    p.h1("References")
    p.references([
        "[1] X. Lu et al., “Slipping trajectory tracking control of wheeled "
        "mobile robot based on dynamics model,” in Proc. IEEE Int. Conf. "
        "Ind. Electron. Appl. (ICIEA), 2024, doi: 10.1109/ICIEA61579.2024."
        "10664745.",
        "[2] R. Fierro and F. L. Lewis, “Control of a nonholonomic mobile "
        "robot: backstepping kinematics into dynamics,” J. Robot. Syst., "
        "vol. 14, no. 3, pp. 149–163, 1997.",
        "[3] T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine, “Soft "
        "actor-critic: off-policy maximum entropy deep reinforcement learning "
        "with a stochastic actor,” in Proc. ICML, 2018.",
        "[4] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "
        "“Proximal policy optimization algorithms,” arXiv:1707.06347, "
        "2017.",
        "[5] S. Fujimoto, H. van Hoof, and D. Meger, “Addressing function "
        "approximation error in actor-critic methods,” in Proc. ICML, 2018.",
        "[6] T. P. Lillicrap et al., “Continuous control with deep "
        "reinforcement learning,” in Proc. ICLR, 2016.",
        "[7] A. Raffin et al., “Stable-Baselines3: reliable reinforcement "
        "learning implementations,” J. Mach. Learn. Res., vol. 22, no. 268, "
        "pp. 1–8, 2021.",
        "[8] R. Amsters and P. Slaets, “TurtleBot 3 as a robotics education "
        "platform,” in Robotics in Education, 2020, pp. 170–181.",
        "[9] D. Wang and C. B. Low, “Modeling and analysis of skidding and "
        "slipping in wheeled mobile robots: control design perspective,” IEEE "
        "Trans. Robot., vol. 24, no. 3, pp. 676–687, 2008.",
        "[10] W.-H. Chen, “Disturbance-observer-based control for nonlinear "
        "systems,” IEEE/ASME Trans. Mechatronics, vol. 9, no. 4, "
        "pp. 706–710, 2004.",
        "[11] F. Künhe, J. Gomes, and W. Fetter, “Mobile robot trajectory "
        "tracking using model predictive control,” in Proc. IEEE Latin-Amer. "
        "Robot. Symp., 2005.",
        "[12] J. Hwangbo et al., “Learning agile and dynamic motor skills for "
        "legged robots,” Sci. Robot., vol. 4, no. 26, eaau5872, 2019.",
        "[13] A. Kendall et al., “Learning to drive in a day,” in Proc. IEEE "
        "Int. Conf. Robot. Autom. (ICRA), 2019, pp. 8248–8254.",
        "[14] T. Johannink et al., “Residual reinforcement learning for robot "
        "control,” in Proc. IEEE Int. Conf. Robot. Autom. (ICRA), 2019, "
        "pp. 6023–6029.",
        "[15] J. Tobin et al., “Domain randomization for transferring deep "
        "neural networks from simulation to the real world,” in Proc. IEEE/RSJ "
        "Int. Conf. Intell. Robots Syst. (IROS), 2017, pp. 23–30.",
    ])
