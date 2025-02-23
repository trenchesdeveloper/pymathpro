# modules/ode_compare.py
import numpy as np
import sympy as sp
from modules import ode_adm_odm

def solve_ode_series(ode_str, y0, order=5, method='ADM'):
    """
    Solve the ODE using a series method (ADM or ODM).
    """
    if method.upper() == 'ADM':
        return ode_adm_odm.solve_ode_adm(ode_str, y0, order)
    elif method.upper() == 'ODM':
        return ode_adm_odm.solve_ode_odm(ode_str, y0, order)
    else:
        raise ValueError("Method must be either ADM or ODM.")

def solve_exact_ode(ode_str, y0):
    """
    Solve the ODE exactly using sympy.dsolve.
    The ODE is given as y' = F(x,y) with initial condition y(0)=y0.
    Returns a simplified expression for y(x).
    """
    x = sp.symbols('x')
    f = sp.Function('f')
    local_dict = {'x': x, 'y': f(x)}
    F = sp.sympify(ode_str, locals=local_dict)
    ode_eq = sp.Eq(sp.diff(f(x), x), F)
    sol = sp.dsolve(ode_eq, f(x), ics={f(0): y0})
    return sp.simplify(sol.rhs)

def evaluate_solution(solution_expr, x_vals):
    """
    Evaluate a sympy solution (series or exact) on a numpy array of x values.
    Returns a numpy array.
    """
    x = sp.symbols('x')
    f = sp.lambdify(x, solution_expr, 'numpy')
    val = f(x_vals)
    if np.isscalar(val):
        val = np.full_like(x_vals, val, dtype=float)
    return val

def compute_error(series_expr, exact_expr, x_vals):
    """
    Compute the absolute error between the series solution and the exact solution over x_vals.
    Returns a numpy array of errors.
    """
    series_vals = evaluate_solution(series_expr, x_vals)
    exact_vals = evaluate_solution(exact_expr, x_vals)
    return np.abs(series_vals - exact_vals)
