import numpy as np
import matplotlib.pyplot as plt

a, b, c, f = 5.0, 1.0, 0.3, 0.9
x0 = 1.0
T = 2.0
dt = 1e-3
N_steps = int(T / dt)
N_paths = 1_000_000
t_arr = np.linspace(0, T, N_steps + 1)

rng = np.random.default_rng(98769876234523423423433333)


def moment_rhs(state):
    m1, sig2, mu3, mu4 = state
    dm1   = -a * m1 + f
    dsig2 = (-2*a + b**2) * sig2 + b**2 * m1**2 + c**2
    dmu3  = 3*(-a + b**2) * mu3 + 6 * b**2 * m1 * sig2
    dmu4  = (-4*a + 6*b**2) * mu4 + 12 * b**2 * m1 * mu3 \
            + 6*(b**2 * m1**2 + c**2) * sig2
    return np.array([dm1, dsig2, dmu3, dmu4])

def rk4_step(state, h):
    k1 = moment_rhs(state)
    k2 = moment_rhs(state + 0.5*h*k1)
    k3 = moment_rhs(state + 0.5*h*k2)
    k4 = moment_rhs(state + h*k3)
    return state + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

state = np.array([x0, 0.0, 0.0, 0.0])
theory = np.zeros((N_steps + 1, 4))
theory[0] = state
for i in range(N_steps):
    state = rk4_step(state, dt)
    theory[i+1] = state

m1_th    = theory[:, 0]
sig2_th  = theory[:, 1]
mu3_th   = theory[:, 2]
mu4_th   = theory[:, 3]

skew_th = np.where(sig2_th > 1e-30, mu3_th / sig2_th**1.5, 0.0)
kurt_th = np.where(sig2_th > 1e-30, mu4_th / sig2_th**2 - 3.0, 0.0)


print(f"Running Euler-Maruyama with {N_paths:,} paths, {N_steps} steps ...")
sqrt_dt = np.sqrt(dt)

x = np.full(N_paths, x0)

mc_mean = np.zeros(N_steps + 1)
mc_var  = np.zeros(N_steps + 1)
mc_mu3  = np.zeros(N_steps + 1)
mc_mu4  = np.zeros(N_steps + 1)
mc_mean[0] = x0

for i in range(N_steps):
    dWb = rng.standard_normal(N_paths) * sqrt_dt
    dWc = rng.standard_normal(N_paths) * sqrt_dt
    x = x + (-a * x + f) * dt + b * x * dWb + c * dWc

    m = np.mean(x)
    dx = x - m
    var = np.mean(dx**2)
    mc_mean[i+1] = m
    mc_var[i+1]  = var
    mc_mu3[i+1]  = np.mean(dx**3)
    mc_mu4[i+1]  = np.mean(dx**4)

mc_skew = np.where(mc_var > 1e-30, mc_mu3 / mc_var**1.5, 0.0)
mc_kurt = np.where(mc_var > 1e-30, mc_mu4 / mc_var**2 - 3.0, 0.0)

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle(
    r"Monte Carlo ($N={:,}$) vs Theoretical Moment ODEs".format(N_paths),
    fontsize=15, fontweight="bold", y=0.98
)

skip = max(1, N_steps // 500)
t_ds = t_arr[::skip]

panels = [
    (axes[0,0], mc_mean[::skip],  m1_th[::skip],   r"Mean $\langle x \rangle$"),
    (axes[0,1], mc_var[::skip],   sig2_th[::skip],  r"Variance $\sigma^2$"),
    (axes[1,0], mc_skew[::skip],  skew_th[::skip],  r"Skewness $\gamma_1 = \mu_3/\sigma^3$"),
    (axes[1,1], mc_kurt[::skip],  kurt_th[::skip],  r"Excess Kurtosis $\kappa = \mu_4/\sigma^4 - 3$"),
]

colors_mc = "#2563eb"
colors_th = "#dc2626"

for ax, mc_data, th_data, title in panels:
    ax.plot(t_ds, mc_data, color=colors_mc, alpha=0.7, linewidth=1.2, label="Monte Carlo")
    ax.plot(t_ds, th_data, color=colors_th, linewidth=2, linestyle="--", label="Theory (ODE)")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(r"$t$", fontsize=11)
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("mc_vs_theory.png", dpi=180, bbox_inches="tight")
plt.show()
