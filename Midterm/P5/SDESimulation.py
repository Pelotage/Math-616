import numpy as np
import matplotlib.pyplot as plt

SEED = 109878909998765434565434
RNG = np.random.default_rng(SEED)

X0 = -94
T = 7.0
DT = 0.005
N_STEPS = int(T / DT)
N_PATHS = 100000
N_SHOW = 60

t = np.linspace(0, T, N_STEPS + 1)

X = np.zeros((N_PATHS, N_STEPS + 1))
X[:, 0] = X0

sqrt_dt = np.sqrt(DT)
for k in range(N_STEPS):
    tk = k * DT
    sigma_k = np.exp(tk)
    dW = sqrt_dt * RNG.standard_normal(N_PATHS)
    X[:, k + 1] = X[:, k] + (-X[:, k]) * DT + sigma_k * dW

mean_theory = X0 * np.exp(-t)
var_theory = (np.exp(2 * t) - np.exp(-2 * t)) / 4

mean_mc = X.mean(axis=0)
var_mc = X.var(axis=0)

fig, axes = plt.subplots(3, 1, figsize=(9, 13))
fig.suptitle(
    r"SDE:  $dX_t = -X_t\,dt + e^{\,t}\,dW_t$,    $X_0 = %.0f$" % X0,
    fontsize=15, fontweight="bold", y=0.995,
)

ax = axes[0]
for i in range(N_SHOW):
    ax.plot(t, X[i], linewidth=0.4, alpha=0.45)
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_title("50 sample paths", fontsize=13)
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$X_t$")

ax = axes[1]
ax.plot(t, mean_mc, color="#534AB7", linewidth=2, label="Monte Carlo mean")
ax.plot(
    t, mean_theory, color="#D85A30", linewidth=2, linestyle="--",
    label=r"Theory: $X_0 e^{-t}$",
)
ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
ax.set_title(r"$\mathbb{E}[X_t]$ converges to 0", fontsize=13)
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$\mathbb{E}[X_t]$")
ax.legend(fontsize=11)

ax = axes[2]
ax.plot(t, var_mc, color="#534AB7", linewidth=2, label="Monte Carlo variance")
ax.plot(
    t, var_theory, color="#D85A30", linewidth=2, linestyle="--",
    label=r"Theory: $\frac{1}{4}(e^{2t} - e^{-2t})$",
)
ax.set_title(r"$\mathrm{Var}(X_t) \to \infty$", fontsize=13)
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$\mathrm{Var}(X_t)$")
ax.legend(fontsize=11)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()
plt.savefig("sde_simulation.png", dpi=180, bbox_inches="tight")
