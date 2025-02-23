# modules/ode_adm_odm.py
import sympy as sp


def solve_ode_adm(ode_str, y0, order=5):
    """
    Solve an ODE using a simplified Adomian Decomposition Method.
    The ODE is given as y' = F(x, y) (with ode_str representing F(x,y)).
    y0 is the initial condition y(0)=y0.
    Returns a series solution (truncated to the specified order) as a sympy expression in x.

    Example:
      For ode_str = "x - y**2" and y0 = 0.
    """
    x, y = sp.symbols('x y')
    # Parse the ODE right-hand side F(x, y)
    F = sp.sympify(ode_str, locals={'x': x, 'y': y})

    # Assume a power series solution:
    #   y(x) = a0 + a1*x + a2*x**2 + ... + a_order*x**order
    coeffs = sp.symbols('a0:' + str(order + 1))
    series_sol = sp.Add(*[coeffs[i] * x ** i for i in range(order + 1)])

    # Compute the derivative of the series solution
    series_diff = sp.diff(series_sol, x)

    # Substitute the series solution into F(x,y) and expand
    F_sub = sp.expand(F.subs(y, series_sol))
    diff_expr = sp.expand(series_diff)

    # Truncate both expressions to order+1 using sp.Add to sum up the coefficients
    diff_truncated = sp.Add(*[diff_expr.coeff(x, i) * x ** i for i in range(order + 1)])
    F_truncated = sp.Add(*[F_sub.coeff(x, i) * x ** i for i in range(order + 1)])

    # Build the system of equations by equating the coefficients for x^i, i=0,...,order
    equations = []
    for i in range(order + 1):
        eq = sp.Eq(diff_truncated.coeff(x, i), F_truncated.coeff(x, i))
        equations.append(eq)

    # Impose the initial condition by forcing a0 = y0 (replace the equation for x^0)
    equations[0] = sp.Eq(coeffs[0], y0)

    sol = sp.solve(equations, list(coeffs), dict=True)
    if not sol:
        raise ValueError("Could not solve for series coefficients.")

    series_solution = sp.expand(series_sol.subs(sol[0]))
    return series_solution


def solve_ode_odm(ode_str, y0, order=5):
    """
    Solve an ODE using a simplified Optimal Decomposition Method.
    (For demonstration purposes, this returns the same series solution as ADM.)
    """
    return solve_ode_adm(ode_str, y0, order)


def evaluate_series(series_expr, x_vals):
    """
    Evaluate the series solution (a sympy expression in x) on a numpy array of x values.
    Returns a numpy array. If the evaluation yields a scalar, it is broadcast to an array.
    """
    x = sp.symbols('x')
    f = sp.lambdify(x, series_expr, 'numpy')
    val = f(x_vals)
    import numpy as np
    if np.isscalar(val):
        val = np.full_like(x_vals, val, dtype=float)
    return val
