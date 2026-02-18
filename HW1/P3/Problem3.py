import numpy as np
import matplotlib.pyplot as plt


def forward_euler_2d(u1_0, u2_0, a, omega, f1, f2, dt, t_max):
    n_steps = int(t_max / dt)
    t = np.linspace(0, t_max, n_steps)

    u1 = np.zeros(n_steps)
    u2 = np.zeros(n_steps)

    u1[0] = u1_0
    u2[0] = u2_0

    for i in range(n_steps - 1):
        du1_dt = -a * u1[i] + omega * u2[i] + f1
        du2_dt = -omega * u1[i] - a * u2[i] + f2

        u1[i+1] = u1[i] + dt * du1_dt
        u2[i+1] = u2[i] + dt * du2_dt

    return u1, u2, t


omega = 1
f1 = 0
f2 = 0
dt = 0.01
t_max = 20
u1_0 = 1
u2_0 = 0

a_values = [0, 1, -1]
labels = ['(i) a = 0', '(ii) a = 1', '(iii) a = -1']

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, (a, label) in enumerate(zip(a_values, labels)):
    u1, u2, t = forward_euler_2d(u1_0, u2_0, a, omega, f1, f2, dt, t_max)

    axes[idx].plot(u1, u2, 'b-', linewidth=1.5)
    axes[idx].plot(u1[0], u2[0], 'go', markersize=8, label='Start')
    axes[idx].plot(u1[-1], u2[-1], 'ro', markersize=8, label='End')
    axes[idx].set_xlabel('$u_1$', fontsize=12)
    axes[idx].set_ylabel('$u_2$', fontsize=12)
    axes[idx].set_title(label, fontsize=13)
    axes[idx].grid(True, alpha=0.3)
    axes[idx].legend()
    axes[idx].axis('equal')

plt.tight_layout()
plt.savefig('trajectories_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
