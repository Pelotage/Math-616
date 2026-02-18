import numpy as np
import matplotlib.pyplot as plt


a = 0.5
f = 2.0
u0 = 5.0
t_final = 10.0


def euler_method(u_0, a, f, dt, t_final):
    n_steps = int(t_final / dt)
    u = u_0
    for i in range(n_steps):
        u = u + dt * (-a*u + f)
    return u


def analytic_method(t, u0, a, f):
    return f/a + (u0 - f/a) * np.exp(-a*t)


dt_values = np.array([1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3])

errors = []
for dt in dt_values:
    u_numerical = euler_method(u0, a, f, dt, t_final)
    u_exact = analytic_method(t_final, u0, a, f)
    error = np.abs(u_numerical - u_exact)
    errors.append(error)
    print(f"dt = {dt:.0e}: u_numerical = {u_numerical:.6f}, u_exact = {
          u_exact:.6f}, error = {error:.6e}")

errors = np.array(errors)

log_dt = np.log10(dt_values)
log_error = np.log10(errors)
slope = np.polyfit(log_dt, log_error, 1)[0]

plt.figure(figsize=(8, 6))
plt.loglog(dt_values, errors, 'bo-', linewidth=2,
           markersize=8, label='Absolute Error')
plt.xlabel(r'Time step $\Delta t$', fontsize=12)
plt.ylabel('Absolute Error', fontsize=12)
plt.title('Error vs Time Step (Forward Euler Method)', fontsize=14)
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.legend(fontsize=11)

plt.text(0.05, 0.95, f'Slope = {slope:.2f}',
         transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('euler_error.png', dpi=300, bbox_inches='tight')
plt.show()
