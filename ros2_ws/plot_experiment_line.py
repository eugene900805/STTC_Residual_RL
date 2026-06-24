#!/usr/bin/env python3
"""Plot the TB3 straight-line slip experiment (slip steps at 30/50 s):
+velocity inner loop vs RL (SAC) vs Residual RL, against the reference path.
Reads /tmp/tb3_exp_line.npz."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

d = np.load('/tmp/tb3_exp_line.npz')
modes = [m for m in ['veloop', 'rl', 'residual'] if f'{m}_t' in d]
LBL = {'veloop': '+ velocity loop', 'rl': 'RL (SAC)', 'residual': 'Residual RL'}
COL = {'veloop': 'C0', 'rl': 'C2', 'residual': 'C3'}


def en(m):
    return np.sqrt(d[f'{m}_xe']**2 + d[f'{m}_ye']**2 + d[f'{m}_the']**2)


def rmse(m):
    e = en(m); k = int(0.7*len(e)); return float(np.mean(e[k:]))


fig, ax = plt.subplots(1, 2, figsize=(12, 5))
if modes and f'{modes[0]}_rx' in d:
    ax[0].plot(d[f'{modes[0]}_rx'], d[f'{modes[0]}_ry'], 'k--', lw=1.5,
               label='reference')
for m in modes:
    ax[0].plot(d[f'{m}_x'], d[f'{m}_y'], COL[m], lw=1.4,
               label=f'{LBL[m]} (RMSE={rmse(m):.3f})')
ax[0].grid(alpha=.3); ax[0].legend()
ax[0].set_xlabel('X [m]'); ax[0].set_ylabel('Y [m]  (lateral deviation)')
ax[0].set_title('TB3 straight line under slip (Gazebo)')

for m in modes:
    ax[1].plot(d[f'{m}_t'], en(m), COL[m], lw=1.3, label=LBL[m])
for tc in (30, 50):
    ax[1].axvline(tc, color='k', ls=':', lw=.8)
ax[1].grid(alpha=.3); ax[1].legend(); ax[1].set_xlabel('t [s]')
ax[1].set_ylabel('|e| [m,rad]'); ax[1].set_title('tracking error (slip steps 30/50 s)')

plt.tight_layout()
plt.savefig('/home/eugene/STT/figures_tb3/slip_all_line.png', dpi=120)
for m in ['nocomp', 'veloop', 'rl', 'residual']:
    if f'{m}_t' in d:
        print(f'{m:9s} steady RMSE = {rmse(m):.4f}')
print('saved figures_tb3/slip_all_line.png')
