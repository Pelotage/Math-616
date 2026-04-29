from sympy import *

p, x = symbols('p x')
l0, l1, l2, l3 = symbols('lambda_0 lambda_1 lambda_2 lambda_3')
integrand = -p*ln(p) - l0*p - l1*x*p - l2*x**2*p - l3*x**3*p

# Functional derivative
func_deriv = diff(integrand, p)
print("Functional derivative:")
print(func_deriv)

# Solve for p
solution = solve(func_deriv, p)
print("\nSolution for p:")
print(solution)
