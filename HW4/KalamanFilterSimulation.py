import numpy as np
import matplotlib.pyplot as plt

a = 1.0
f = 1.0
sigma_x = 1.0
dt = 0.4
sigma_o = 0.5
T = 500.0

N = int(T / dt)
np.random.seed(2147895)

A = np.exp(a * dt)
B = (f / a) * (np.exp(a * dt) - 1.0)
R_f = (sigma_x**2 / (2 * a)) * (np.exp(2 * a * dt) - 1.0)
R_o = sigma_o**2

x_true = np.zeros(N + 1)
x_true[0] = 0.0

for k in range(N):
    x_true[k + 1] = A * x_true[k] + B + np.sqrt(R_f) * np.random.randn()

y_obs = x_true[1:] + sigma_o * np.random.randn(N)

x_filter = np.zeros(N + 1)
P_filter = np.zeros(N + 1)


x_filter[0] = 0.0
P_filter[0] = 1.0

for k in range(N):
    x_pred = A * x_filter[k] + B
    P_pred = A**2 * P_filter[k] + R_f

    K = P_pred / (P_pred + R_o)
    x_filter[k + 1] = x_pred + K * (y_obs[k] - x_pred)
    P_filter[k + 1] = (1.0 - K) * P_pred

residuals = x_true - x_filter

T_show = 25.0
N_show = int(T_show / dt)

t_grid = np.arange(N + 1) * dt
t_obs = np.arange(1, N + 1) * dt

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                         gridspec_kw={'height_ratios': [3, 1, 1]})

ax = axes[0]
std_band = 2 * np.sqrt(P_filter)
ax.fill_between(t_grid[:N_show+1],
                x_filter[:N_show+1] - std_band[:N_show+1],
                x_filter[:N_show+1] + std_band[:N_show+1],
                color='dodgerblue', alpha=0.20, label=r'Filter $\pm 2\sigma$')
ax.plot(t_grid[:N_show+1], x_true[:N_show+1], 'k-',
        lw=0.8, alpha=0.7, label='True signal')
ax.scatter(t_obs[:N_show], y_obs[:N_show], s=12, c='red',
           alpha=0.5, zorder=3, label='Observations')
ax.plot(t_grid[:N_show+1], x_filter[:N_show+1],
        'b-', lw=1.2, label='Kalman filter mean')
ax.set_ylabel(r'$x(t)$', fontsize=13)
ax.set_title(
    rf'Kalman Filter:  $a={a}$, $f={f}$, $\sigma_x={sigma_x}$, '
    rf'$\Delta t={dt}$, $\sigma_o={sigma_o}$   (showing first {int(T_show)} of {
        int(T)} time units)',
    fontsize=13)
ax.legend(fontsize=10, loc='upper left')
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(t_grid[:N_show+1], residuals[:N_show+1], 'k-', lw=0.6)
ax.axhline(0, color='grey', ls='--', lw=0.5)
ax.set_ylabel('Error', fontsize=13)
ax.set_title('Filter error  (true − filter mean)', fontsize=11)
ax.grid(True, alpha=0.3)

ax = axes[2]
ax.plot(t_grid[:N_show+1], P_filter[:N_show+1], 'b-', lw=1.0,
        label=r'Posterior variance $P_{k|k}$')
ax.set_ylabel(r'$P_{k|k}$', fontsize=13)
ax.set_xlabel(r'Time $t$', fontsize=13)
ax.set_title('Posterior variance (converges to steady state)', fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('kalman_filter_results.png',
            dpi=150, bbox_inches='tight')
plt.show()
