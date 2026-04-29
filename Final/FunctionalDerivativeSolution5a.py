from sympy import *

p, x, l0, l1, l2 = symbols('p x lambda_0 lambda_1 lambda_2')

integrand = -p*ln(p) - l0*p - l1*x*p - l2*x**2*p

# Functional derivative
func_deriv = diff(integrand, p)
print("Functional derivative:")
print(func_deriv)

# Solve for p
solution = solve(func_deriv, p)
print("\nSolution for p:")
print(solution)
