import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

np.random.seed(38246588)

dt = 0.001
sigma = 1.0
x0 = 0.0


def simulate_sde(a, x0, dt, sigma, T):
    N = int(T / dt)
    X = np.zeros(N + 1)
    X[0] = x0
    dW = np.sqrt(dt) * np.random.randn(N)
    for i in range(N):
        X[i + 1] = X[i] + a * X[i] * dt + sigma * dW[i]
        X[i + 1] = np.clip(X[i + 1], -1e6, 1e6)
    return X


def run_ensemble(a, x0, dt, sigma, T, n_traj):
    endpoints = np.zeros(n_traj)
    for k in range(n_traj):
        traj = simulate_sde(a, x0, dt, sigma, T)
        endpoints[k] = traj[-1]
    return endpoints


a_unstable = 1.0
T_unstable = 4.0
n_ensemble = 5000

ensemble_unstable = run_ensemble(
    a_unstable, x0, dt, sigma, T_unstable, n_ensemble)

T_long_unstable = 4.0
single_traj_unstable = simulate_sde(a_unstable, x0, dt, sigma, T_long_unstable)
time_samples_unstable = single_traj_unstable[len(single_traj_unstable) // 2:]

a_stable = -2.0
T_stable = 10.0
n_ensemble_stable = 5000

ensemble_stable = run_ensemble(
    a_stable, x0, dt, sigma, T_stable, n_ensemble_stable)

T_long_stable = 200.0
single_traj_stable = simulate_sde(a_stable, x0, dt, sigma, T_long_stable)
time_samples_stable = single_traj_stable[len(single_traj_stable) // 2:]

var_theory = sigma**2 / (2 * abs(a_stable))
x_theory = np.linspace(-3, 3, 500)
pdf_theory = (1 / np.sqrt(2 * np.pi * var_theory)) * \
    np.exp(-x_theory**2 / (2 * var_theory))

fig = plt.figure(figsize=(14, 10))
gs = GridSpec(2, 2, hspace=0.35, wspace=0.3)

ax1 = fig.add_subplot(gs[0, 0])
t_grid = np.arange(0, T_unstable + dt, dt)[:len(single_traj_unstable)]
ax1.plot(t_grid, single_traj_unstable,
         linewidth=0.4, color='#d62728', alpha=0.8)
for _ in range(4):
    traj = simulate_sde(a_unstable, x0, dt, sigma, T_unstable)
    ax1.plot(t_grid[:len(traj)], traj, linewidth=0.4, alpha=0.5)
ax1.set_title(f'Unstable SDE: $a = +{a_unstable}$  (sample trajectories)',
              fontsize=12, fontweight='bold')
ax1.set_xlabel('$t$')
ax1.set_ylabel('$X_t$')
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(gs[0, 1])

lo = np.percentile(np.concatenate(
    [ensemble_unstable, time_samples_unstable]), 1)
hi = np.percentile(np.concatenate(
    [ensemble_unstable, time_samples_unstable]), 99)
bins = np.linspace(lo, hi, 80)

ax2.hist(ensemble_unstable, bins=bins, density=True, alpha=0.55,
         color='#1f77b4', label=f'Ensemble at $T={T_unstable}$', edgecolor='white', linewidth=0.3)
ax2.hist(time_samples_unstable, bins=bins, density=True, alpha=0.55,
         color='#d62728', label='Time average (single traj)', edgecolor='white', linewidth=0.3)
ax2.set_title('Unstable: PDFs do NOT match', fontsize=12, fontweight='bold')
ax2.set_xlabel('$x$')
ax2.set_ylabel('Density')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(gs[1, 0])
t_grid_s = np.arange(0, T_long_stable + dt, dt)[:len(single_traj_stable)]
ax3.plot(t_grid_s[:5000], single_traj_stable[:5000],
         linewidth=0.4, color='#2ca02c', alpha=0.8)
ax3.set_title(f'Stable SDE: $a = {a_stable}$  (sample trajectory)',
              fontsize=12, fontweight='bold')
ax3.set_xlabel('$t$')
ax3.set_ylabel('$X_t$')
ax3.grid(True, alpha=0.3)

ax4 = fig.add_subplot(gs[1, 1])
lo_s = np.percentile(np.concatenate(
    [ensemble_stable, time_samples_stable]), 0.5)
hi_s = np.percentile(np.concatenate(
    [ensemble_stable, time_samples_stable]), 99.5)
bins_s = np.linspace(lo_s, hi_s, 80)

ax4.hist(ensemble_stable, bins=bins_s, density=True, alpha=0.55,
         color='#1f77b4', label=f'Ensemble at $T={T_stable}$', edgecolor='white', linewidth=0.3)
ax4.hist(time_samples_stable, bins=bins_s, density=True, alpha=0.55,
         color='#2ca02c', label='Time average (single traj)', edgecolor='white', linewidth=0.3)
ax4.plot(x_theory, pdf_theory, 'k--', linewidth=2,
         label=r'Theory: $\mathcal{N}(0,\,\sigma^2/2|a|)$')
ax4.set_title('Stable: PDFs match (ergodic)', fontsize=12, fontweight='bold')
ax4.set_xlabel('$x$')
ax4.set_ylabel('Density')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

plt.savefig('sde_comparison.png', dpi=150, bbox_inches='tight',
            facecolor='white')
plt.show()
