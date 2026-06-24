#!/usr/bin/env python3
"""Fig. 8 -- algorithm vs algorithm+RL-residual on the TB3 under injected slip.
Base velocity inner loop vs base + small learned residual, against the reference
path (with a zoom inset on the top arc).  Reads /tmp/tb3_exp.npz."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

d = np.load('/tmp/tb3_exp.npz')
BASE, RES = 'veloop', 'residual'
CB, CG = 'C0', 'C2'


def en(m):
    return np.sqrt(d[f'{m}_xe']**2 + d[f'{m}_ye']**2 + d[f'{m}_the']**2)


def rmse(m):
    e = en(m); k = int(0.7*len(e)); return float(np.mean(e[k:]))


rb, rr = rmse(BASE), rmse(RES)
ratio = rb / rr

fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.78))

# --- left: trajectory + reference, with zoom inset on the top arc ---
ax[0].plot(d[f'{BASE}_rx'], d[f'{BASE}_ry'], 'k--', lw=1.6, label='reference')
ax[0].plot(d[f'{BASE}_x'], d[f'{BASE}_y'], CB, lw=1.5,
           label=f'base velocity loop (RMSE = {rb:.4f} m)')
ax[0].plot(d[f'{RES}_x'], d[f'{RES}_y'], CG, lw=1.5,
           label=f'base + RL residual  (RMSE = {rr:.4f} m)')
ax[0].axis('equal'); ax[0].grid(alpha=.3); ax[0].legend(loc='lower left',
                                                         fontsize=8)
ax[0].set_xlabel('X [m]'); ax[0].set_ylabel('Y [m]')
ax[0].set_title('TurtleBot3 circle under slip: algorithm vs. algorithm + RL '
                'residual', fontsize=9)

# zoom inset on the top of the circle
axin = ax[0].inset_axes([0.60, 0.60, 0.38, 0.34])
axin.plot(d[f'{BASE}_rx'], d[f'{BASE}_ry'], 'k--', lw=1.4)
axin.plot(d[f'{BASE}_x'], d[f'{BASE}_y'], CB, lw=1.4)
axin.plot(d[f'{RES}_x'], d[f'{RES}_y'], CG, lw=1.4)
x0, x1, y0, y1 = -0.30, 0.32, 1.24, 1.46
axin.set_xlim(x0, x1); axin.set_ylim(y0, y1)
axin.set_xticks([]); axin.set_yticks([])
ax[0].indicate_inset_zoom(axin, edgecolor='gray')
ax[0].annotate('zoom', xy=(-0.05, 1.40), xytext=(-0.95, 1.15), fontsize=8,
               color='gray',
               arrowprops=dict(arrowstyle='->', color='gray', lw=.8))

# --- right: tracking error with the slip steps + ratio annotation ---
ax[1].plot(d[f'{BASE}_t'], en(BASE), CB, lw=1.4, label='base velocity loop')
ax[1].plot(d[f'{RES}_t'], en(RES), CG, lw=1.4, label='base + RL residual')
for tc in (30, 50):
    ax[1].axvline(tc, color='k', ls=':', lw=.8)
ax[1].grid(alpha=.3); ax[1].legend(loc='upper left', fontsize=8)
ax[1].set_xlabel('t [s]'); ax[1].set_ylabel('|e| [m, rad]')
ax[1].set_title('tracking error  (slip steps at 30 s, 50 s)', fontsize=9)
ax[1].annotate(f'{ratio:.1f}× lower steady error\nwith RL residual',
               xy=(46, en(RES)[np.argmin(np.abs(d[f'{RES}_t']-46))]),
               xytext=(33, 0.040), fontsize=8,
               bbox=dict(boxstyle='round', fc='#fff3cd', ec='none'),
               arrowprops=dict(arrowstyle='->', color='gray', lw=.8))

plt.tight_layout()
plt.savefig('/home/eugene/STT/figures_tb3/residual_vs_base.png', dpi=120)
print(f'base   steady RMSE = {rb:.4f}')
print(f'res    steady RMSE = {rr:.4f}')
print(f'ratio  = {ratio:.2f}x')
print('saved figures_tb3/residual_vs_base.png')
