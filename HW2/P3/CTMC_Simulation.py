import numpy as np
import matplotlib.pyplot as plt

s1 = -0.04
s2 = 2.27

alpha = 1.5
beta = 3.0

rng = np.random.default_rng(seed=9937465927475)


def simulate_ctmc(alpha, beta, T_total=5000.0, initial_state=0):
    rates = [alpha, beta]
    state = initial_state
    t = 0.0
    jump_times = [0.0]
    state_seq = [state]
    hold_0 = []
    hold_1 = []

    while t < T_total:
        rate = rates[state]
        dt = rng.exponential(1.0 / rate)

        if t + dt > T_total:
            dt = T_total - t

        if state == 0:
            hold_0.append(dt)
        else:
            hold_1.append(dt)

        t += dt
        if t >= T_total:
            break

        state = 1 - state
        jump_times.append(t)
        state_seq.append(state)

    return (np.array(jump_times),
            np.array(state_seq),
            [np.array(hold_0), np.array(hold_1)])


T_total = 50_000.0
jump_times, state_seq, hold_times = simulate_ctmc(alpha, beta, T_total)

durations = np.diff(np.append(jump_times, T_total))
pi1_hat = durations[state_seq == 0].sum() / T_total
pi2_hat = durations[state_seq == 1].sum() / T_total
state_values = np.where(state_seq == 0, s1, s2)
EX_hat = (state_values * durations).sum() / T_total
mean_wait_s1 = hold_times[0].mean()
mean_wait_s2 = hold_times[1].mean()

pi1_theory = beta / (alpha + beta)
pi2_theory = alpha / (alpha + beta)
EX_theory = pi1_theory * s1 + pi2_theory * s2
mean_wait_s1_theory = 1.0 / alpha
mean_wait_s2_theory = 1.0 / beta

print(f"Simulation length : {T_total:.0f} time units")
print(f"Number of jumps   : {len(jump_times) - 1}")

print("\n──(b) Estimates──────────────────────")
print(f"  s1  : {pi1_hat:.6f}")
print(f"  s2  : {pi2_hat:.6f}")
print(f"  E[X]                   : {EX_hat:.6f}")
print(f"  Mean wait in s1        : {mean_wait_s1:.6f}")
print(f"  Mean wait in s2        : {mean_wait_s2:.6f}")

print("\n──(c) Theoretical───────────────────────")
print(f"  s1  : {pi1_theory:.6f}")
print(f"  s2  : {pi2_theory:.6f}")
print(f"  E[X]                   : {EX_theory:.6f}")
print(f"  Mean wait in s1        : {mean_wait_s1_theory:.6f}")
print(f"  Mean wait in s2        : {mean_wait_s2_theory:.6f}")

print("\n──Relative Error──────────────────────────────")
print(f"  s1  : {abs(pi1_hat - pi1_theory)/pi1_theory * 100:.4f} %")
print(f"  s2  : {abs(pi2_hat - pi2_theory)/pi2_theory * 100:.4f} %")
print(f"  E[X]: {abs(EX_hat - EX_theory) / abs(EX_theory) * 100:.4f} %")
print(f"  Mean s1  : {abs(mean_wait_s1 - mean_wait_s1_theory) /
      mean_wait_s1_theory * 100:.4f} %")
print(f"  Mean s2  : {abs(mean_wait_s2 - mean_wait_s2_theory) /
      mean_wait_s2_theory * 100:.4f} %")

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("Two-State Continuous-Time Markov Jump Process",
             fontsize=15, fontweight='bold')

ax = axes[0, 0]
plot_end = min(50.0, jump_times[-1])
mask = jump_times <= plot_end
t_plot = np.append(jump_times[mask], plot_end)
s_plot = np.append(state_seq[mask], state_seq[mask][-1])
x_plot = np.where(s_plot == 0, s1, s2)
ax.step(t_plot, x_plot, where='post', color='steelblue', linewidth=1.5)
ax.axhline(s1, color='tomato',   linestyle='--',
           alpha=0.6, label=f'$s_1={s1}$')
ax.axhline(s2, color='seagreen', linestyle='--',
           alpha=0.6, label=f'$s_2={s2}$')
ax.set_xlabel('Time $t$')
ax.set_ylabel('$X_t$')
ax.set_title('Sample Path (first 50 time units)')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
cum_time_s1 = np.cumsum(np.where(state_seq == 0, durations, 0.0))
running_pi1 = cum_time_s1 / np.maximum(jump_times + durations, 1e-12)
idx = np.linspace(0, len(running_pi1)-1, 2000, dtype=int)
ax.plot(jump_times[idx], running_pi1[idx], color='steelblue',
        linewidth=1.2, label='Running $\\hat{\\pi}_1$')
ax.axhline(pi1_theory, color='tomato', linestyle='--',
           linewidth=1.5, label=f'Theory $\\pi_1={pi1_theory:.4f}$')
ax.set_xlabel('Time $t$')
ax.set_ylabel('Estimated $\\pi_1$')
ax.set_title('Convergence of $\\hat{\\pi}_1$ to Stationary Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
bins = 60
t_exp = np.linspace(0, max(hold_times[0].max(), hold_times[1].max()), 300)
ax.hist(hold_times[0], bins=bins, density=True, alpha=0.55,
        color='steelblue', label='State $s_1$ sojourns')
ax.hist(hold_times[1], bins=bins, density=True, alpha=0.55,
        color='seagreen',  label='State $s_2$ sojourns')
ax.plot(t_exp, alpha * np.exp(-alpha * t_exp), 'steelblue',
        linewidth=2, label=f'Exp($\\alpha$={alpha}) PDF')
ax.plot(t_exp, beta * np.exp(-beta * t_exp), 'seagreen',
        linewidth=2, label=f'Exp($\\beta$={beta}) PDF')
ax.set_xlabel('Holding time')
ax.set_ylabel('Density')
ax.set_title('Holding-Time Distributions')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
labels = ['$\\pi_1$', '$\\pi_2$',
          '$\\mathbb{E}[X]/3$', '$\\tau_1$', '$\\tau_2$']
num_vals = [pi1_hat, pi2_hat, EX_hat/3, mean_wait_s1, mean_wait_s2]
thy_vals = [pi1_theory, pi2_theory, EX_theory /
            3, mean_wait_s1_theory, mean_wait_s2_theory]
x_pos = np.arange(len(labels))
width = 0.35
ax.bar(x_pos - width/2, num_vals, width,
       label='Simulation', color='steelblue', alpha=0.8)
ax.bar(x_pos + width/2, thy_vals, width,
       label='Theory',     color='tomato',    alpha=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel('Value')
ax.set_title(
    'Simulation vs. Theory\n($\\mathbb{E}[X]$ scaled by 1/3 for display)')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('markov_jump_process.png',
            dpi=150, bbox_inches='tight')
plt.show()
