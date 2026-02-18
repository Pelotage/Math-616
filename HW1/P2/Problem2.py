import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Fixed parameters
b = -4
c = 4


def f_dynamics(t, u, a, f_param):
    return a*u + b*u**2 - c*u**3 + f_param


def find_all_roots(a, f_param):
    coeffs = [-c, b, a, f_param]
    roots = np.roots(coeffs)
    return roots


def stability(u, a, f_param):
    return a + 2*b*u - 3*c*u**2


def create_comprehensive_plot(a, f_param, initial_conditions, case_name):
    roots = find_all_roots(a, f_param)
    real_roots = sorted([r.real for r in roots if np.abs(r.imag) < 1e-10])

    fig = plt.figure(figsize=(14, 6))

    # Plot 1: Time Evolution
    ax1 = plt.subplot(1, 2, 1)
    colors = plt.cm.tab10(np.linspace(0, 1, len(initial_conditions)))

    for i, u0 in enumerate(initial_conditions):
        sol = solve_ivp(f_dynamics, (0, 10), [u0], args=(a, f_param),
                        dense_output=True, max_step=0.01)
        ax1.plot(sol.t, sol.y[0], label=f'$u_0 = {
                 u0}$', color=colors[i], linewidth=2)

    for root in real_roots:
        ax1.axhline(y=root, color='red', linestyle='--',
                    alpha=0.5, linewidth=1.5)

    ax1.set_xlabel('$t$', fontsize=12)
    ax1.set_ylabel('$u(t)$', fontsize=12)
    ax1.set_title(f'{case_name}\nTime Evolution',
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(1, 2, 2)
    u_range = np.linspace(-3, 3, 1000)
    dudt = a*u_range + b*u_range**2 - c*u_range**3 + f_param
    ax2.plot(u_range, dudt, 'b-', linewidth=2.5, label='$du/dt$')
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=1)
    ax2.axvline(x=0, color='k', linestyle='-', alpha=0.3, linewidth=1)

    for i, root in enumerate(roots):
        if np.abs(root.imag) < 1e-10:  # Real root
            dfdu = stability(root.real, a, f_param)
            if dfdu < 0:
                marker = 'o'
                color = 'green'
                stab = 'stable'
            else:
                marker = 's'
                color = 'red'
                stab = 'unstable'

            ax2.plot(root.real, 0, marker, color=color, markersize=12,
                     markeredgecolor='black', markeredgewidth=1.5,
                     label=f'$r_{i+1} = {root.real:.4f}$ ({stab})', zorder=5)

    ax2.set_xlabel('$u$', fontsize=12)
    ax2.set_ylabel('$du/dt$', fontsize=12)
    ax2.set_title('Phase Portrait (○ = stable, □ = unstable)',
                  fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([-2.5, 2.5])
    ax2.set_ylim([-50, 75])

    plt.tight_layout()
    return fig, roots


a1, f1 = 4, 10
initial_conditions1 = [-2, 0, 1, 1.3, 2]
fig1, roots1 = create_comprehensive_plot(a1, f1, initial_conditions1,
                                         f'Case 1: $a = {a1}, f = {f1}$')

a2, f2 = 4, 2
initial_conditions2 = [-2, -1.5, -0.5, 0, 0.8, 1.5]
fig2, roots2 = create_comprehensive_plot(a2, f2, initial_conditions2,
                                         f'Case 2: $a = {a2}, f = {f2}$')


fig1.savefig('case1_a4_f10_analysis.png',
             dpi=150, bbox_inches='tight')
fig2.savefig('case2_a4_f2_analysis.png',
             dpi=150, bbox_inches='tight')

plt.show()
