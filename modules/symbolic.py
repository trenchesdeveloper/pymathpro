# modules/symbolic.py
import sympy as sp
import ast


def parse_expression(expr_str, var_str="x"):
    """
    Parse a string into a sympy expression.
    Supports multiple variables if var_str contains commas.
    """
    try:
        # Allow multiple variables if a comma is present.
        if ',' in var_str:
            variables = sp.symbols(var_str)
        else:
            variables = sp.symbols(var_str)
        expr = sp.sympify(expr_str)
    except sp.SympifyError as e:
        raise ValueError(f"Could not parse expression '{expr_str}'. Ensure it is valid.") from e
    return expr, variables


def differentiate(expr_str, var_str="x"):
    expr, var = parse_expression(expr_str, var_str)
    derivative = sp.diff(expr, var)
    return derivative


def integrate(expr_str, var_str="x"):
    expr, var = parse_expression(expr_str, var_str)
    integral = sp.integrate(expr, var)
    return integral


def taylor_series(expr_str, var_str="x", point=0, order=10):
    expr, var = parse_expression(expr_str, var_str)
    series = sp.series(expr, var, point, order).removeO()
    return series


def simplify_expr(expr_str):
    expr, _ = parse_expression(expr_str)
    return sp.simplify(expr)


def compute_limit(expr_str, var_str="x", point=0):
    expr, var = parse_expression(expr_str, var_str)
    result = sp.limit(expr, var, point)
    return result


def solve_equation(equation_str, var_str="x"):
    """
    Solve a single equation symbolically.
    If the input includes '=', it is split into left and right parts.
    """
    if "=" in equation_str:
        left_str, right_str = equation_str.split("=")
        left_expr, _ = parse_expression(left_str, var_str)
        right_expr, _ = parse_expression(right_str, var_str)
        eq = sp.Eq(left_expr, right_expr)
    else:
        expr, _ = parse_expression(equation_str, var_str)
        eq = sp.Eq(expr, 0)
    var = sp.symbols(var_str)
    solutions = sp.solve(eq, var)
    return solutions


def factor_expr(expr_str):
    expr, _ = parse_expression(expr_str)
    return sp.factor(expr)


def expand_expr(expr_str):
    expr, _ = parse_expression(expr_str)
    return sp.expand(expr)


def partial_fraction(expr_str, var_str="x"):
    """
    Compute the partial fraction decomposition of a rational function.
    """
    expr, _ = parse_expression(expr_str, var_str)
    return sp.apart(expr)


def solve_system(equations_str, variables_str="x"):
    """
    Solve a system of equations symbolically.
    Equations should be separated by semicolons (;) or newlines.
    """
    try:
        equations_list = [eq.strip() for eq in equations_str.replace('\n', ';').split(';') if eq.strip()]
        eqs = []
        for eq_str in equations_list:
            if "=" in eq_str:
                left_str, right_str = eq_str.split("=")
                eq = sp.Eq(sp.sympify(left_str), sp.sympify(right_str))
            else:
                eq = sp.Eq(sp.sympify(eq_str), 0)
            eqs.append(eq)
        if ',' in variables_str:
            vars = sp.symbols(variables_str)
        else:
            vars = sp.symbols(variables_str)
        if not isinstance(vars, (tuple, list)):
            vars = [vars]
        solutions = sp.solve(eqs, vars)
        return solutions
    except Exception as e:
        raise ValueError("Error solving system: " + str(e))


def solve_ode(ode_str, func_str, var_str="x"):
    """
    Solve an ordinary differential equation symbolically.
    Example: ode_str = "f(x).diff(x) - f(x)", func_str = "f", var_str = "x"
    """
    try:
        x = sp.symbols(var_str)
        f = sp.Function(func_str)
        if "=" in ode_str:
            left_str, right_str = ode_str.split("=")
            left_expr = sp.sympify(left_str, locals={func_str: f})
            right_expr = sp.sympify(right_str, locals={func_str: f})
            ode = sp.Eq(left_expr, right_expr)
        else:
            ode = sp.Eq(sp.sympify(ode_str, locals={func_str: f}), 0)
        solution = sp.dsolve(ode, f)
        return solution
    except Exception as e:
        raise ValueError("Error solving ODE: " + str(e))


def parse_matrix(matrix_str):
    """
    Parse a string representing a matrix into a sympy Matrix.
    Example: "[[1, 2], [3, 4]]"
    """
    try:
        matrix_list = ast.literal_eval(matrix_str)
        return sp.Matrix(matrix_list)
    except Exception as e:
        raise ValueError("Error parsing matrix: " + str(e))


def matrix_determinant(matrix_str):
    M = parse_matrix(matrix_str)
    return M.det()


def matrix_inverse(matrix_str):
    M = parse_matrix(matrix_str)
    return M.inv()


def matrix_eigen(matrix_str):
    M = parse_matrix(matrix_str)
    eigen_data = M.eigenvects()  # Returns list of (eigenvalue, multiplicity, [eigenvectors])
    return eigen_data


def generate_plot(expr_str, result_expr=None, operation=None, var_str='x'):
    """
    Generate an interactive Plotly graph of the original function.
    If a computed result is provided (for differentiation, integration, or Taylor series),
    it will be overlaid on the same plot.
    """
    import numpy as np
    import plotly.graph_objects as go
    import plotly.io as pio

    x = sp.symbols(var_str)
    orig_expr, _ = parse_expression(expr_str, var_str)
    f_orig = sp.lambdify(x, orig_expr, "numpy")

    x_vals = np.linspace(-10, 10, 400)
    try:
        y_orig = f_orig(x_vals)
    except Exception as e:
        raise ValueError("Error evaluating the original function for plotting: " + str(e))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_vals, y=y_orig, mode='lines', name='Original Function'))

    if result_expr is not None and operation in ['differentiate', 'integrate', 'taylor']:
        f_result = sp.lambdify(x, result_expr, "numpy")
        try:
            y_result = f_result(x_vals)
        except Exception as e:
            raise ValueError("Error evaluating the computed result for plotting: " + str(e))
        trace_name = {
            'differentiate': "Derivative",
            'integrate': "Antiderivative",
            'taylor': "Taylor Series Approximation"
        }.get(operation, "Computed Result")
        fig.add_trace(go.Scatter(x=x_vals, y=y_result, mode='lines', name=trace_name))

    fig.update_layout(title="Function Plot", xaxis_title='x', yaxis_title='y')
    return pio.to_html(fig, full_html=False)
