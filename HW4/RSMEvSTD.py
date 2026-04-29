import numpy as np
import matplotlib.pyplot as plt

a = 1.0
f = 1.0
sigma_x = 1.0
dt = 0.4
T = 500.0
N = int(T / dt)

A = np.exp(a * dt)
B = (f / a) * (np.exp(a * dt) - 1.0)
R_f = (sigma_x**2 / (2 * a)) * (np.exp(2 * a * dt) - 1.0)

t_start_eval = 50.0
k_start = int(t_start_eval / dt)

sigma_o_vals = np.linspace(0.01, 2.0, 200)

N_mc = 50

np.random.seed(2147895)

rmse_mc = np.zeros(len(sigma_o_vals))

for i, sigma_o in enumerate(sigma_o_vals):
    R_o = sigma_o**2
    sum_sq_err = 0.0
    count = 0

    for mc in range(N_mc):
        e = 0.0
        P = 1.0

        sq_err_sum_this = 0.0

        for k in range(N):
            w_k = np.random.randn()
            v_k = np.random.randn()

            P_pred = A**2 * P + R_f
            K = P_pred / (P_pred + R_o)
            P = (1.0 - K) * P_pred

            e = (1.0 - K) * (A * e + np.sqrt(R_f) * w_k) \
                - K * sigma_o * v_k

            if k + 1 >= k_start:
                sq_err_sum_this += e**2

        n_eval = N - k_start + 1
        sum_sq_err += sq_err_sum_this
        count += n_eval

    rmse_mc[i] = np.sqrt(sum_sq_err / count)

rmse_analytical = np.zeros(len(sigma_o_vals))

for i, sigma_o in enumerate(sigma_o_vals):
    R_o = sigma_o**2
    P = 1.0
    for _ in range(200):
        P_pred = A**2 * P + R_f
        P = P_pred * R_o / (P_pred + R_o)
    rmse_analytical[i] = np.sqrt(P)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(sigma_o_vals, rmse_mc, 'b-', lw=2,
        label='Monte Carlo RMSE')
ax.plot(sigma_o_vals, rmse_analytical, 'r--', lw=2,
        label=r'Analytical $\sqrt{P_{\infty}}$')
ax.set_xlabel(r'Observational noise $\sigma_o$', fontsize=14)
ax.set_ylabel('RMSE of posterior mean', fontsize=14)
ax.set_title(
    r'Kalman Filter RMSE vs $\sigma_o$'
    rf'  ($a={a}$, $f={f}$, $\sigma_x={sigma_x}$, $\Delta t={dt}$)'
    f'\nEvaluated over $t \\in [{int(t_start_eval)},\\, {int(T)}]$'
    f',  {N_mc} Monte Carlo realizations',
    fontsize=13)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_xlim([sigma_o_vals[0], sigma_o_vals[-1]])
ax.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig('rmse_vs_sigma_o.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved.")

print(f"\n{'sigma_o':>10s}  {'RMSE (MC)':>12s}  {'RMSE (theory)':>14s}")
print("-" * 42)
for s in [0.01, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0]:
    idx = np.argmin(np.abs(sigma_o_vals - s))
    print(f"{sigma_o_vals[idx]:10.3f}  {rmse_mc[idx]:12.4f}  "
          f"{rmse_analytical[idx]:14.4f}")
