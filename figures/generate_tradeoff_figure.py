import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

unsw_alpha = [0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
unsw_p20   = [58.1, 62.4, 64.7, 67.9, 70.1, 69.6]
unsw_speedup = [18.8, 55.9, 126.6, 327.6, 530.0, 598.8]
unsw_baseline_p20 = 43.4

cic_alpha = [0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
cic_p20   = [33.7, 32.7, 38.2, 48.0, 55.1, 55.1]
cic_speedup = [14.5, 32.1, 65.0, 116.0, 180.9, 171.5]
cic_baseline_p20 = 36.2

ton_alpha = [0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
ton_p20   = [38.4, 38.7, 44.1, 52.8, 60.4, 60.7]
ton_speedup = [11.9, 27.5, 57.4, 126.8, 205.4, 197.5]
ton_baseline_p20 = 38.6

color_unsw = '#0072B2'
color_cic  = '#D55E00'
color_ton  = '#009E73'

fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.6))

ax.plot(unsw_speedup, unsw_p20, 'o-', color=color_unsw, label='UNSW-NB15', markersize=6, linewidth=1.8, zorder=3)
ax.plot(cic_speedup, cic_p20, 's-', color=color_cic, label='CICIDS2017 (named service)', markersize=6, linewidth=1.8, zorder=3)
ax.plot(ton_speedup, ton_p20, '^-', color=color_ton, label='TON_IoT (no temporal signal)', markersize=6.5, linewidth=1.8, zorder=3)

ax.axhline(unsw_baseline_p20, color=color_unsw, linestyle=':', linewidth=1.1, alpha=0.55, zorder=1)
ax.axhline(cic_baseline_p20, color=color_cic, linestyle=':', linewidth=1.1, alpha=0.55, zorder=1)
ax.axhline(ton_baseline_p20, color=color_ton, linestyle=':', linewidth=1.1, alpha=0.55, zorder=1)

ax.text(460, unsw_baseline_p20 + 1.2, 'UNSW-NB15 full corr.', color=color_unsw, fontsize=7, alpha=0.85, ha='right')
ax.text(460, ton_baseline_p20 + 1.2, 'TON_IoT full corr.', color=color_ton, fontsize=7, alpha=0.85, ha='right')
ax.text(460, cic_baseline_p20 - 2.6, 'CICIDS2017 full corr.', color=color_cic, fontsize=7, alpha=0.85, ha='right')

for x, y, a in zip(unsw_speedup, unsw_p20, unsw_alpha):
    if a == 0.01:
        ax.annotate(f'$\\alpha$={a}', (x, y), textcoords="offset points", xytext=(6, 6), fontsize=7.5, color=color_unsw)
for x, y, a in zip(cic_speedup, cic_p20, cic_alpha):
    if a == 0.01:
        ax.annotate(f'$\\alpha$={a}', (x, y), textcoords="offset points", xytext=(8, -12), fontsize=7.5, color=color_cic)
for x, y, a in zip(ton_speedup, ton_p20, ton_alpha):
    if a == 0.005:
        ax.annotate(f'$\\alpha$={a}', (x, y), textcoords="offset points", xytext=(-8, 8), fontsize=7.5, color=color_ton, ha='right')

ax.set_xscale('log')
ax.set_xlabel('Speedup over full correlation (log scale)')
ax.set_ylabel('Precision@20 (%)')
ax.set_title('Precision/Speedup Trade-off Across Selectivity Levels')
ax.legend(loc='upper left', frameon=False, fontsize=8.5)
ax.grid(True, which='both', alpha=0.2)
ax.set_ylim(28, 75)

plt.tight_layout()
plt.savefig('/tmp/figure_tradeoffs_v3.pdf', dpi=300, bbox_inches='tight')
plt.savefig('/tmp/figure_tradeoffs_v3.png', dpi=200, bbox_inches='tight')
print("Saved")
