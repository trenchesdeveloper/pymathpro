# modules/numerical.py
import numpy as np

def safe_eval(func_str):
    """
    Convert a string into a lambda function for numerical evaluation.
    The input should be a valid expression in terms of x (and y for ODEs).
    Only names from numpy (and variables x, y) are allowed.
    """
    allowed_names = {k: getattr(np, k) for k in dir(np) if not k.startswith("__")}
    allowed_names['x'] = None
    allowed_names['y'] = None
    try:
        code = compile(func_str, "<string>", "eval")
        for name in code.co_names:
            if name not in allowed_names:
                raise ValueError(f"Use of '{name}' is not allowed.")
        return lambda **kwargs: eval(code, {"__builtins__": {}}, {**allowed_names, **kwargs})
    except Exception as e:
        raise ValueError("Error in function expression: " + str(e))

def numerical_derivative(func_str, x, h=1e-5):
    """
    Approximate the derivative f'(x) using the central difference formula.
    """
    f = safe_eval(func_str)
    return (f(x=x + h) - f(x=x - h)) / (2 * h)

def trapezoidal_integration(func_str, a, b, n):
    """
    Approximate the integral of f(x) from a to b using the trapezoidal rule.
    n: number of subintervals.
    """
    f = safe_eval(func_str)
    x_vals = np.linspace(a, b, n + 1)
    y_vals = f(x=x_vals)
    h = (b - a) / n
    integral = h * (0.5 * y_vals[0] + 0.5 * y_vals[-1] + np.sum(y_vals[1:-1]))
    return integral

def rk4_solver(func_str, x0, y0, h, steps):
    """
    Solve the ODE y' = f(x, y) using the 4th-order Runge-Kutta (RK4) method.
    Returns two arrays: x values and y values.
    """
    f = safe_eval(func_str)
    xs = [x0]
    ys = [y0]
    x, y = x0, y0
    for i in range(steps):
        k1 = f(x=x, y=y)
        k2 = f(x=x + h/2, y=y + h/2*k1)
        k3 = f(x=x + h/2, y=y + h/2*k2)
        k4 = f(x=x + h, y=y + h*k3)
        y += (h/6) * (k1 + 2*k2 + 2*k3 + k4)
        x += h
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def newton_raphson(func_str, x0, tol=1e-5, max_iter=100):
    """
    Find a root of f(x)=0 using the Newton-Raphson method.
    """
    f = safe_eval(func_str)
    for i in range(max_iter):
        fx = f(x=x0)
        dfx = (f(x=x0 + tol) - f(x=x0 - tol)) / (2 * tol)
        if abs(dfx) < tol:
            raise ValueError("Derivative too small; method fails.")
        x_new = x0 - fx / dfx
        if abs(x_new - x0) < tol:
            return x_new
        x0 = x_new
    raise ValueError("Newton-Raphson did not converge within the maximum number of iterations.")
