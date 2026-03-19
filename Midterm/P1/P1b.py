import numpy as np
import matplotlib.pyplot as plt

u0 = np.array([0.5, 1.5])
T = 5.0
dt = 0.01
N = int(T / dt)

c1 = -1.0 / 6.0
c2 = -1.0 / 3.0

t_exact = np.linspace(0, T, 500)
u1_exact = 1 + c1 * np.exp(-t_exact) + c2 * np.exp(-4 * t_exact)
u2_exact = 1 + c1 * np.exp(-t_exact) - 2 * c2 * np.exp(-4 * t_exact)


def f(u):
    u1, u2 = u
    return np.array([
        -2*u1 + u2 + 1,
        2*u1 - 3*u2 + 1
    ])


t_euler = np.zeros(N + 1)
u_euler = np.zeros((N + 1, 2))
u_euler[0] = u0

for i in range(N):
    t_euler[i + 1] = t_euler[i] + dt
    u_euler[i + 1] = u_euler[i] + dt * f(u_euler[i])

print(f"{'t':>6}  {'u1 (analytic)':>14}  {'u1 (Euler)':>12}  "
      f"{'u2 (analytic)':>14}  {'u2 (Euler)':>12}")
print("-" * 70)

sample_times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
for t_s in sample_times:
    idx = round(t_s / dt)
    u1_a = 1 + c1 * np.exp(-t_s) + c2 * np.exp(-4 * t_s)
    u2_a = 1 + c1 * np.exp(-t_s) - 2 * c2 * np.exp(-4 * t_s)
    u1_n, u2_n = u_euler[idx]
    print(f"{t_s:>6.1f}  {u1_a:>14.6f}  {
          u1_n:>12.6f}  {u2_a:>14.6f}  {u2_n:>12.6f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
fig.suptitle("Forward Euler vs Analytic Solution",
             fontsize=13, fontweight="normal")

labels = [r"$u_1(t)$", r"$u_2(t)$"]
exact = [u1_exact, u2_exact]
euler = [u_euler[:, 0], u_euler[:, 1]]
colors = ["#185FA5", "#0F6E56"]

for ax, lbl, ex, eu, col in zip(axes, labels, exact, euler, colors):
    ax.plot(t_exact,  ex, color=col,   lw=2,   label="Analytic")
    ax.plot(t_euler,  eu, color=col,   lw=1.2, ls="--",
            alpha=0.8, label=f"Forward Euler ($\\Delta t={dt}$)")
    ax.set_xlabel("$t$", fontsize=12)
    ax.set_ylabel(lbl,   fontsize=12)
    ax.set_title(lbl,    fontsize=12, fontweight="normal")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("Forward_Euler.png",
            dpi=150, bbox_inches="tight")
plt.show()
