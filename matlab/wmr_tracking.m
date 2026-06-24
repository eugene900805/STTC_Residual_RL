function wmr_tracking()
% WMR_TRACKING  Slipping trajectory tracking control of a wheeled mobile robot.
%
%   Reproduction of:
%   C. Lu et al., "Slipping Trajectory Tracking Control of Wheeled Mobile
%   Robot Based on Dynamics Model," IEEE ICIEA 2024.
%
%   Implements the full hierarchical scheme of the paper:
%     * kinematic backstepping virtual control law (eq.12)
%     * online slip-ratio adaptive estimator (eq.13/14)
%     * dynamic backstepping torque law (eq.18) on the wheel-rotation dynamics
%       (eq.7: I_w*alpha_ddot = tau - r*Fx)
%
%   Runs the two scenarios of the paper (straight line, circle) with the
%   left/right slip parameters stepping at t = 50 s and t = 30 s, and
%   reproduces Fig.4-9 (trajectory, tracking error, slip-rate estimation),
%   plus an adaptive-vs-no-compensation comparison and the dynamic-layer run.
%
%   Just run:  >> wmr_tracking
%   Figures are saved as PNG into ./figures_matlab/

    clc; close all;
    outdir = 'figures_matlab';
    if ~exist(outdir, 'dir'); mkdir(outdir); end

    P = params();

    fprintf('WMR slipping trajectory tracking (paper reproduction)\n');
    for traj = {'line', 'circle'}
        name = traj{1};

        % --- kinematic backstepping + adaptive slip estimation (paper core) ---
        log_on  = simulate_wmr(name, P, true,  false);
        % --- baseline: no slip compensation (i fixed at 1) ---
        log_off = simulate_wmr(name, P, false, false);
        % --- full dynamic (torque) cascade + adaptive slip estimation ---
        log_dyn = simulate_wmr(name, P, true,  true);

        fprintf(['  [%-6s] steady-state RMSE:  kin+adapt = %.4e | ' ...
                 'no-comp = %.4e | dynamic = %.4e\n'], ...
                 name, steady_rmse(log_on), steady_rmse(log_off), ...
                 steady_rmse(log_dyn));

        plot_results(log_on,  name, outdir, '');
        plot_compare(log_on,  log_off, name, outdir);
        plot_results(log_dyn, name, outdir, 'dyn');
    end
    fprintf('Done. Figures saved in ./%s/\n', outdir);
end


%% ======================================================================= %%
%  Parameters (paper Section V)
%% ======================================================================= %%
function P = params()
    % geometry
    P.b = 0.15;     % half distance between rear wheels [m]
    P.d = 0.2;      % centroid offset from drive-wheel axle [m]
    P.r = 0.125;    % wheel radius [m]
    % kinematic controller / adaptive gains (eq.12-14)
    P.Hx  = 5.0;
    P.Hy  = 5.0;
    P.Hs  = 0.2;
    P.lam = 0.5;    % lambda in [0,1]
    P.rho1 = 10.0;  % adaptation gain, left wheel
    P.rho2 = 4.0;   % adaptation gain, right wheel
    % inertial parameters (dynamic layer)
    P.m   = 10.0;   % body mass [kg]
    P.m_w = 0.5;    % one drive wheel + rotor [kg]
    P.I   = 0.5;    % body inertia about G [kg m^2]
    P.I_w = 0.05;   % wheel inertia about its axle [kg m^2]
    % dynamic layer numerics / gains
    P.Ct    = 120.0;  % linear traction stiffness [N s]
    P.n_sub = 40;     % fine substeps per control step (contact is stiff)
    P.c1    = 800.0;  % wheel-speed inner-loop gain (re-tuned, see notes)
    P.tau_f = 0.05;   % LP filter on body w fed back to the kinematic layer
    % effective body inertia in (v, w) space
    P.m_eff = P.m + 2*P.m_w + 2*P.I_w/P.r^2;
    P.I_eff = P.I + 2*P.m_w*P.b^2 + 2*P.I_w*P.b^2/P.r^2;
    % simulation
    P.T  = 60.0;
    P.dt = 0.01;
end


%% ======================================================================= %%
%  Reference trajectories (paper Section V)
%% ======================================================================= %%
function [pose, vr, wr] = trajectory(name, t)
    switch name
        case 'line'
            % x = x0 + vr t cos(a), y = y0 + vr t sin(a); constant heading
            x0 = 1.8; y0 = 0.2; a = pi/6; vr = 2.0; wr = 0.0;
            pose = [x0 + vr*t*cos(a); y0 + vr*t*sin(a); a];
        case 'circle'
            % x = R cos(wr t), y = R sin(wr t); tangent heading
            R = 2.0; wr = 1.0; vr = R*wr;
            pose = [R*cos(wr*t); R*sin(wr*t); wr*t + pi/2];
        otherwise
            error('unknown trajectory: %s', name);
    end
end

function pose0 = initial_pose(name)
    % robot starts offset from the reference to show convergence
    switch name
        case 'line';   pose0 = [1.6; 0.0; pi/4];
        case 'circle'; pose0 = [1.5; -0.4; 0.0];
    end
end


%% ======================================================================= %%
%  True longitudinal slip ratios s_L(t), s_R(t)
%  (paper: parameters change at t = 30 s and t = 50 s)
%% ======================================================================= %%
function [sL, sR] = slip_profile(t)
    sL0 = 0.10; sR0 = 0.10;     % initial slip
    sL1 = 0.25; sR1 = 0.20;     % after the step
    if t < 30.0, sR = sR0; else, sR = sR1; end
    if t < 50.0, sL = sL0; else, sL = sL1; end
end


%% ======================================================================= %%
%  Kinematic relations
%% ======================================================================= %%
function [v, w] = wheel_to_body(wL, wR, sL, sR, P)
    % effective ground velocity of each wheel (eq.1): v_k = r w_k (1 - s_k)
    vL = P.r * wL * (1 - sL);
    vR = P.r * wR * (1 - sR);
    v = 0.5 * (vR + vL);
    w = (vR - vL) / (2 * P.b);
end

function dq = centroid_kinematics(theta, v, w, d)
    % velocity of centroid G (offset d ahead of the axle along heading)
    dq = [v*cos(theta) - d*w*sin(theta);
          v*sin(theta) + d*w*cos(theta);
          w];
end

function e = pose_error(pose, ref)
    % tracking error in the robot body frame (reference - actual)
    th = pose(3);
    dx = ref(1) - pose(1);
    dy = ref(2) - pose(2);
    xe =  cos(th)*dx + sin(th)*dy;
    ye = -sin(th)*dx + cos(th)*dy;
    the = atan2(sin(ref(3)-th), cos(ref(3)-th));   % wrapped
    e = [xe; ye; the];
end


%% ======================================================================= %%
%  Controller: kinematic virtual control law + adaptive slip estimation
%% ======================================================================= %%
function [vc, wc] = virtual_velocity(e, vr, wr, w_meas, P)
    % eq.12
    xe = e(1); ye = e(2); the = e(3);
    vc = vr*cos(the) + P.Hx*(xe + P.d*(1-cos(the))) - P.Hs*the*w_meas;
    wc = wr + vr*( P.Hy*(1-P.lam)*(ye - P.d*sin(the) + P.Hs*the) ...
                   + (P.lam/P.Hs)*sin(the) );
end

function [iL, iR] = adapt(e, vc, wc, iL, iR, P, dt)
    % adaptive slip-estimation laws (eq.13/14), Euler integration.
    % estimate i_k = 1/(1 - s_k) (eq.9)
    xe = e(1); ye = e(2); the = e(3);
    He2 = (1/P.Hy)*sin(the);
    He1 = P.Hs*(ye - P.d*sin(the) + xe*the + P.d*the*(1-cos(the)) + P.Hs*the);
    base = P.b*xe + P.b*P.d*(1-cos(the));
    iR_dot = (1/(2*P.b)) * P.rho2 * (vc + P.b*wc) * (base + He1 + He2);
    iL_dot = (1/(2*P.b)) * P.rho1 * (vc - P.b*wc) * (base - He1 - He2);
    iR = iR + iR_dot*dt;
    iL = iL + iL_dot*dt;
    % keep estimates physically meaningful (i = 1/(1-s) bounded away from 0)
    iL = min(max(iL, 0.2), 10.0);
    iR = min(max(iR, 0.2), 10.0);
end

function [wL, wR] = wheel_commands(vc, wc, iL, iR, P)
    % map virtual velocity to wheel angular velocities with i-estimates (eq.11)
    wL = (iL/P.r) * (vc - P.b*wc);
    wR = (iR/P.r) * (vc + P.b*wc);
end


%% ======================================================================= %%
%  Simulation loop.  use_dynamics = false -> kinematic plant (slip applied
%  directly to wheel commands).  use_dynamics = true -> dynamic torque cascade:
%  the inner loop drives wheel speed to the slip-compensated command with
%  torque (eq.7/18), integrated on a fine substep so the time-scale separation
%  the hierarchical design assumes is preserved.
%% ======================================================================= %%
function log = simulate_wmr(name, P, adaptive, use_dynamics)
    dt = P.dt; n = round(P.T/dt);
    pose = initial_pose(name);
    iL = 1.0; iR = 1.0;          % start assuming no slip

    % plant states
    v_body = 0.0; w_body = 0.0;  % body velocities
    wL_st = 0.0;  wR_st = 0.0;   % wheel angular velocities (dynamic only)
    w_fb  = 0.0;  w_filt = 0.0;  % body w fed back to the kinematic layer

    log = init_log(n);

    for i = 1:n
        t = (i-1)*dt;
        [ref, vr, wr] = trajectory(name, t);
        e = pose_error(pose, ref);

        [vc, wc] = virtual_velocity(e, vr, wr, w_fb, P);
        if adaptive
            [iL, iR] = adapt(e, vc, wc, iL, iR, P, dt);
        end
        [wLc, wRc] = wheel_commands(vc, wc, iL, iR, P);

        [sL, sR] = slip_profile(t);

        if ~use_dynamics
            % --- kinematic plant: slip degrades the commanded wheel speeds ---
            [v_body, w_body] = wheel_to_body(wLc, wRc, sL, sR, P);
            th = pose(3);
            k1 = centroid_kinematics(th,                v_body, w_body, P.d);
            k2 = centroid_kinematics(th + 0.5*dt*k1(3), v_body, w_body, P.d);
            k3 = centroid_kinematics(th + 0.5*dt*k2(3), v_body, w_body, P.d);
            k4 = centroid_kinematics(th + dt*k3(3),     v_body, w_body, P.d);
            pose = pose + (dt/6)*(k1 + 2*k2 + 2*k3 + k4);
            w_fb = w_body;
        else
            % --- dynamic torque cascade on a fine substep (ZOH on wLc,wRc) ---
            h = dt / P.n_sub;
            for s = 1:P.n_sub
                % inner-loop backstepping torque (eq.18, wheel-speed form)
                tauL = P.I_w * P.c1 * (wLc - wL_st);
                tauR = P.I_w * P.c1 * (wRc - wR_st);
                % wheel rotational dynamics: I_w wdot = tau - r Fx (eq.7)
                FxL = P.Ct*(P.r*wL_st*(1-sL) - (v_body - P.b*w_body))/P.r;
                FxR = P.Ct*(P.r*wR_st*(1-sR) - (v_body + P.b*w_body))/P.r;
                wL_st = wL_st + (tauL - P.r*FxL)/P.I_w * h;
                wR_st = wR_st + (tauR - P.r*FxR)/P.I_w * h;
                % body dynamics driven by the same traction forces
                v_body = v_body + (FxL + FxR)/P.m_eff * h;
                w_body = w_body + (P.b*(FxR - FxL))/P.I_eff * h;
                % integrate centroid pose
                pose = pose + h*centroid_kinematics(pose(3), v_body, w_body, P.d);
            end
            % LP-filtered body w fed back to the kinematic layer next step
            a = dt/(P.tau_f + dt);
            w_filt = w_filt + a*(w_body - w_filt);
            w_fb = w_filt;
        end

        % log
        log.t(i)  = t;
        log.x(i)  = pose(1);  log.y(i)  = pose(2);
        log.xr(i) = ref(1);   log.yr(i) = ref(2);
        log.xe(i) = e(1);     log.ye(i) = e(2);  log.the(i) = e(3);
        log.sLt(i)= sL;       log.sRt(i)= sR;
        log.sLh(i)= 1 - 1/iL; log.sRh(i)= 1 - 1/iR;
    end
end

function log = init_log(n)
    log.t  = zeros(1,n);
    log.x  = zeros(1,n);  log.y  = zeros(1,n);
    log.xr = zeros(1,n);  log.yr = zeros(1,n);
    log.xe = zeros(1,n);  log.ye = zeros(1,n);  log.the = zeros(1,n);
    log.sLt= zeros(1,n);  log.sRt= zeros(1,n);
    log.sLh= zeros(1,n);  log.sRh= zeros(1,n);
end

function v = steady_rmse(log)
    k = round(0.8*numel(log.t)):numel(log.t);
    v = sqrt(mean(log.xe(k).^2 + log.ye(k).^2 + log.the(k).^2));
end


%% ======================================================================= %%
%  Plotting (reproduces Fig.4-9).  tag '' for kinematic, 'dyn' for dynamic.
%% ======================================================================= %%
function plot_results(log, name, outdir, tag)
    if isempty(tag), sfx = ''; ttl = ''; else, sfx = ['_' tag]; ttl = [' (' tag ')']; end

    % 1) trajectory in the plane
    f = figure('Name',[name ' trajectory' ttl],'Color','w');
    plot(log.xr, log.yr, 'k--', 'LineWidth', 2); hold on;
    plot(log.x,  log.y,  'b-',  'LineWidth', 1.5);
    plot(log.x(1), log.y(1), 'go', 'MarkerFaceColor','g');
    axis equal; grid on;
    xlabel('X [m]'); ylabel('Y [m]');
    title(sprintf('WMR %s trajectory tracking%s', name, ttl));
    legend('reference','actual','start','Location','best');
    saveas(f, fullfile(outdir, sprintf('%s_trajectory%s.png', name, sfx)));

    % 2) tracking error
    f = figure('Name',[name ' error' ttl],'Color','w');
    plot(log.t, log.xe, 'LineWidth',1.2); hold on;
    plot(log.t, log.ye, 'LineWidth',1.2);
    plot(log.t, log.the,'LineWidth',1.2);
    plot([log.t(1) log.t(end)], [0 0], 'k-'); grid on;
    xlabel('t [s]'); ylabel('error');
    title(sprintf('WMR %s tracking error%s', name, ttl));
    legend('x_e','y_e','\theta_e','Location','best');
    saveas(f, fullfile(outdir, sprintf('%s_error%s.png', name, sfx)));

    % 3) slip-rate estimation
    f = figure('Name',[name ' slip' ttl],'Color','w');
    plot(log.t, log.sLt, '--', 'Color',[0 0.4 0.8],'LineWidth',1.5); hold on;
    plot(log.t, log.sLh, '-',  'Color',[0 0.4 0.8],'LineWidth',1.5);
    plot(log.t, log.sRt, '--', 'Color',[0.8 0.1 0.1],'LineWidth',1.5);
    plot(log.t, log.sRh, '-',  'Color',[0.8 0.1 0.1],'LineWidth',1.5);
    grid on; xlabel('t [s]'); ylabel('slip ratio');
    title(sprintf('WMR %s slip-rate estimation%s', name, ttl));
    legend('s_L true','s_L est','s_R true','s_R est','Location','best');
    saveas(f, fullfile(outdir, sprintf('%s_slip%s.png', name, sfx)));
end

function plot_compare(log_on, log_off, name, outdir)
    en  = sqrt(log_on.xe.^2  + log_on.ye.^2  + log_on.the.^2);
    eff = sqrt(log_off.xe.^2 + log_off.ye.^2 + log_off.the.^2);
    f = figure('Name',[name ' compare'],'Color','w');
    plot(log_off.t, eff, 'r', 'LineWidth',1.3); hold on;
    plot(log_on.t,  en,  'b', 'LineWidth',1.3);
    yl = ylim;
    plot([30 30], yl, 'k:'); plot([50 50], yl, 'k:'); grid on;
    xlabel('t [s]'); ylabel('||e||');
    title(sprintf('%s: tracking error, adaptive vs. no compensation', name));
    legend('no compensation','adaptive (paper)','Location','best');
    saveas(f, fullfile(outdir, sprintf('%s_compare.png', name)));
end
