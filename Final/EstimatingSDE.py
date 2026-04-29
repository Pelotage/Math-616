import numpy as np
import matplotlib.pyplot as plt

np.random.seed(986531621)

a_true = -1.0
b_true = 1.0
phi_true = np.pi
c_true = 0.5
sigma_true = 0.5

T = 500.0
dt_sim = 0.01
N_sim = int(T / dt_sim)
t_sim = np.linspace(0, T, N_sim + 1)

x_sim = np.zeros(N_sim + 1)
x_sim[0] = 0.1

for i in range(N_sim):
    t_i = t_sim[i]
    xi = x_sim[i]
    drift = a_true * xi + b_true * \
        np.sin(2 * np.pi * t_i + phi_true) * xi**2 - np.exp(c_true) * xi**3
    dW = np.sqrt(dt_sim) * np.random.randn()
    x_sim[i + 1] = xi + drift * dt_sim + sigma_true * dW

dt_obs = 0.1
obs_indices = np.arange(0, N_sim + 1, int(dt_obs / dt_sim))
t_obs = t_sim[obs_indices]
x_obs = x_sim[obs_indices]
N_obs = len(t_obs)

alpha_true = b_true * np.sin(phi_true)
beta_true = b_true * np.cos(phi_true)
gamma_true = np.exp(c_true)

n_aug = 5
H = np.zeros((1, n_aug))
H[0, 0] = 1.0
n_sub = int(dt_obs / dt_sim)
dt_sub = dt_sim
R_obs = np.array([[1e-6]])


def run_ekf_pass(x_obs, t_obs, N_obs, z_init, P_init, Q_diag, sigma_est):
    z_est = z_init.copy()
    P_est = P_init.copy()

    z_history = np.zeros((N_obs, n_aug))
    P_history = np.zeros((N_obs, n_aug, n_aug))
    sigma_history = np.zeros(N_obs)
    z_history[0] = z_est.copy()
    P_history[0] = P_est.copy()
    sigma_history[0] = sigma_est

    innov_sq_sum = 0.0

    for k in range(1, N_obs):
        z_pred = z_est.copy()
        P_pred = P_est.copy()

        for s in range(n_sub):
            t_s = t_obs[k - 1] + s * dt_sub
            xk = z_pred[0]
            ak = z_pred[1]
            alphak = z_pred[2]
            betak = z_pred[3]
            gammak = z_pred[4]

            s2 = np.sin(2 * np.pi * t_s)
            c2 = np.cos(2 * np.pi * t_s)
            trig = alphak * c2 + betak * s2

            f_x = ak * xk + trig * xk**2 - gammak * xk**3

            F = np.zeros((n_aug, n_aug))
            F[0, 0] = ak + 2 * trig * xk - 3 * gammak * xk**2
            F[0, 1] = xk
            F[0, 2] = c2 * xk**2
            F[0, 3] = s2 * xk**2
            F[0, 4] = -xk**3

            z_pred[0] = xk + f_x * dt_sub
            Phi = np.eye(n_aug) + F * dt_sub
            G = np.zeros((n_aug, n_aug))
            G[0, 0] = sigma_est**2
            P_pred = Phi @ P_pred @ Phi.T + G * dt_sub

        Q_add = np.zeros((n_aug, n_aug))
        np.fill_diagonal(Q_add[1:, 1:], Q_diag)
        P_pred += Q_add

        S_val = (H @ P_pred @ H.T + R_obs)[0, 0]
        K = (P_pred @ H.T) / S_val
        innovation = x_obs[k] - z_pred[0]
        z_est = z_pred + K.flatten() * innovation
        P_est = (np.eye(n_aug) - K @ H) @ P_pred
        P_est = 0.5 * (P_est + P_est.T)

        innov_sq_sum += innovation**2

        z_history[k] = z_est.copy()
        P_history[k] = P_est.copy()
        sigma_history[k] = sigma_est

    sigma_final = np.sqrt(innov_sq_sum / (N_obs - 1) / dt_obs)

    return z_history, P_history, sigma_history, z_est, P_est, sigma_final


z_init = np.array([x_obs[0], 0.0, 0.0, 1.0, 1.0])
P_init = np.diag([0.01, 2.0, 2.0, 2.0, 2.0])
Q_diag = np.array([1e-4, 1e-4, 1e-4, 1e-4])
sigma_est = 1.0

for p in range(5):
    z_history, P_history, sigma_history, z_final, P_final, sigma_final = run_ekf_pass(
        x_obs, t_obs, N_obs, z_init, P_init, Q_diag, sigma_est
    )
    z_init = z_final.copy()
    z_init[0] = x_obs[0]
    P_init = np.diag([0.01, 0.5, 0.5, 0.5, 0.5])
    sigma_est = sigma_final
    Q_diag = Q_diag * 0.5

a_final = z_history[-1, 1]
alpha_final = z_history[-1, 2]
beta_final = z_history[-1, 3]
gamma_final = z_history[-1, 4]

x_ekf = z_history[:, 0]
residuals = np.zeros(N_obs - 1)
for k in range(N_obs - 1):
    xk = x_ekf[k]
    t_k = t_obs[k]
    s2 = np.sin(2 * np.pi * t_k)
    c2 = np.cos(2 * np.pi * t_k)
    drift_k = a_final * xk + \
        (alpha_final * c2 + beta_final * s2) * xk**2 - gamma_final * xk**3
    residuals[k] = x_obs[k + 1] - x_obs[k] - drift_k * dt_obs

sigma_running = np.zeros(N_obs)
sigma_running[0] = np.nan
cumsum_r2 = 0.0
for k in range(N_obs - 1):
    cumsum_r2 += residuals[k]**2
    sigma_running[k + 1] = np.sqrt(cumsum_r2 / (k + 1) / dt_obs)

sigma_qv = sigma_running[-1]

a_est = a_final
alpha_est = alpha_final
beta_est = beta_final
gamma_est = gamma_final

b_est = np.sqrt(alpha_est**2 + beta_est**2)
phi_est = np.arctan2(alpha_est, beta_est)
if phi_est < 0:
    phi_est += 2 * np.pi
c_est = np.log(max(gamma_est, 1e-10))

param_names = [r'a', r'b', r'phi', r'c', r'sigma']
true_values = [a_true, b_true, phi_true, c_true, sigma_true]
est_values = [a_est, b_est, phi_est, c_est, sigma_qv]

print("Parameter Estimation Results (Reparametrized EKF):")
print("-" * 55)
for i in range(5):
    print(f"{param_names[i]:>10s}: True = {true_values[i]:8.4f}, "
          f"Est = {est_values[i]:8.4f}")

reparam_names = [r'$a$', r'$\alpha$', r'$\beta$', r'$\gamma$']
reparam_true = [a_true, alpha_true, beta_true, gamma_true]

fig, axes = plt.subplots(6, 1, figsize=(12, 16), sharex=True)

axes[0].plot(t_obs, x_obs, 'k.', markersize=1, label='Observed')
axes[0].plot(t_obs, z_history[:, 0], 'r-', linewidth=0.5, label='EKF Estimate')
axes[0].set_ylabel(r'$x(t)$')
axes[0].legend(loc='upper right')
axes[0].set_title(
    'State and Parameter Estimation via Reparametrized EKF (5-Pass)')

for i in range(4):
    idx = 1 + i
    est = z_history[:, idx]
    std = np.sqrt(P_history[:, idx, idx])
    axes[i + 1].plot(t_obs, est, 'b-', linewidth=0.8)
    axes[i + 1].axhline(y=reparam_true[i], color='r',
                        linestyle='--', linewidth=0.8)
    axes[i + 1].set_ylabel(reparam_names[i])

axes[5].plot(t_obs, sigma_running, 'b-', linewidth=0.8)
axes[5].axhline(y=sigma_true, color='r', linestyle='--', linewidth=0.8)
axes[5].set_ylabel(r'$\sigma$')

axes[-1].set_xlabel(r'$t$')
plt.tight_layout()
plt.savefig('ekf_sde_results.png', dpi=150, bbox_inches='tight')
plt.show()
