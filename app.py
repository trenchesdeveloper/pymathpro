# app.py
from flask import Flask, render_template, request
import numpy as np
import logging
from modules import symbolic, numerical, ode_compare  # Import both modules

# Configure logging for debugging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/symbolic', methods=['GET', 'POST'])
def symbolic_page():
    # [Symbolic module route as defined earlier...]
    result = None
    error = None
    plot_html = None

    if request.method == 'POST':
        expression = request.form.get('expression')
        operation = request.form.get('operation')
        plot_option = request.form.get('plot')
        variables = request.form.get('variables') or "x"

        try:
            if operation == 'differentiate':
                result = symbolic.differentiate(expression, variables)
            elif operation == 'integrate':
                result = symbolic.integrate(expression, variables)
            elif operation == 'taylor':
                order = request.form.get('order')
                order_int = int(order) if order and order.isdigit() else 10
                result = symbolic.taylor_series(expression, variables, order=order_int)
            elif operation == 'simplify':
                result = symbolic.simplify_expr(expression)
            elif operation == 'limit':
                limit_point = request.form.get('limit_point')
                try:
                    point_value = float(limit_point)
                except (ValueError, TypeError):
                    point_value = 0
                result = symbolic.compute_limit(expression, variables, point=point_value)
            elif operation == 'solve':
                result = symbolic.solve_equation(expression, variables)
            elif operation == 'partial_fraction':
                result = symbolic.partial_fraction(expression, variables)
            elif operation == 'factor':
                result = symbolic.factor_expr(expression)
            elif operation == 'expand':
                result = symbolic.expand_expr(expression)
            elif operation == 'solve_system':
                system_equations = request.form.get('system_equations')
                system_variables = request.form.get('system_variables') or "x"
                result = symbolic.solve_system(system_equations, system_variables)
            elif operation == 'solve_ode':
                ode_function = request.form.get('ode_function')
                ode_indep = request.form.get('ode_indep') or "x"
                result = symbolic.solve_ode(expression, ode_function, ode_indep)
            elif operation == 'matrix_determinant':
                matrix = request.form.get('matrix')
                result = symbolic.matrix_determinant(matrix)
            elif operation == 'matrix_inverse':
                matrix = request.form.get('matrix')
                result = symbolic.matrix_inverse(matrix)
            elif operation == 'matrix_eigen':
                matrix = request.form.get('matrix')
                result = symbolic.matrix_eigen(matrix)
            else:
                error = "Invalid operation selected."
        except Exception as e:
            error = str(e)

        if plot_option and operation in ['differentiate', 'integrate', 'taylor']:
            try:
                first_var = variables.split(',')[0].strip()
                plot_html = symbolic.generate_plot(expression, result, operation, var_str=first_var)
            except Exception as e:
                error = "Plotting error: " + str(e)

    return render_template('symbolic.html', result=result, error=error, plot_html=plot_html)

# app.py (excerpt from numerical_page route)
@app.route('/numerical', methods=['GET', 'POST'])
def numerical_page():
    result = None
    error = None
    plot_html = None
    xs = None
    ys = None

    if request.method == 'POST':
        method = request.form.get('method')
        func_str = request.form.get('func')
        try:
            if method == 'differentiate':
                x_val_str = request.form.get('x_val')
                if x_val_str == "" or x_val_str is None:
                    raise ValueError("Point x is required for differentiation.")
                x_val = float(x_val_str)
                h = float(request.form.get('h') or 1e-5)
                result = numerical.numerical_derivative(func_str, x_val, h)
            elif method == 'integrate':
                a = float(request.form.get('a'))
                b = float(request.form.get('b'))
                n = int(request.form.get('n'))
                result = numerical.trapezoidal_integration(func_str, a, b, n)
            elif method == 'rk4':
                x0 = float(request.form.get('x0_ode'))
                y0 = float(request.form.get('y0'))
                h = float(request.form.get('h_ode'))
                steps = int(request.form.get('steps'))
                xs, ys = numerical.rk4_solver(func_str, x0, y0, h, steps)
                result = "ODE solution computed. See graph below."
                import plotly.graph_objects as go
                import plotly.io as pio
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', name='ODE Solution'))
                fig.update_layout(title="ODE Solution via RK4", xaxis_title='x', yaxis_title='y')
                plot_html = pio.to_html(fig, full_html=False)
            elif method == 'newton':
                x0_str = request.form.get('x0_newton')
                if x0_str == "" or x0_str is None:
                    raise ValueError("Initial guess (x0) is required for Newton-Raphson.")
                x0 = float(x0_str)
                result = numerical.newton_raphson(func_str, x0)
            else:
                error = "Invalid numerical method selected."
        except Exception as e:
            error = str(e)
    return render_template('numerical.html', result=result, error=error, plot_html=plot_html, xs=xs, ys=ys)


@app.route('/odecompare', methods=['GET', 'POST'])
def odecompare_page():
    result = None
    plot_html = None
    conclusion = None
    error_msg = None
    if request.method == 'POST':
        ode_str = request.form.get('ode_str')
        method_type = request.form.get('method_type')  # 'ADM' or 'ODM'
        try:
            y0 = float(request.form.get('y0'))
        except Exception as e:
            error_msg = "Initial condition y0 must be a number."
            return render_template('ode_compare.html', error=error_msg)
        try:
            order = int(request.form.get('order') or 5)
        except Exception as e:
            order = 5
        try:
            x_start = float(request.form.get('x_start') or 0)
            x_end = float(request.form.get('x_end') or 1)
            num_points = int(request.form.get('num_points') or 100)
        except Exception as e:
            x_start, x_end, num_points = 0, 1, 100

        try:
            # Get series solution using selected method
            series_solution = ode_compare.solve_ode_series(ode_str, y0, order, method_type)
            # Get exact solution using sympy.dsolve
            exact_solution = ode_compare.solve_exact_ode(ode_str, y0)

            # Evaluate solutions over the specified x-range
            x_vals = np.linspace(x_start, x_end, num_points)
            series_vals = ode_compare.evaluate_solution(series_solution, x_vals)
            exact_vals = ode_compare.evaluate_solution(exact_solution, x_vals)

            # Compute average error
            errors = ode_compare.compute_error(series_solution, exact_solution, x_vals)
            avg_error = np.mean(errors)

            tol = 1e-3
            if avg_error < tol:
                conclusion = f"The series solution using {method_type} converges well (avg error = {avg_error:.2e})."
            else:
                conclusion = f"The series solution using {method_type} does not match closely (avg error = {avg_error:.2e})."

            # Create an interactive Plotly graph comparing the two solutions
            import plotly.graph_objects as go
            import plotly.io as pio
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_vals, y=series_vals, mode='lines', name='Series Solution'))
            fig.add_trace(go.Scatter(x=x_vals, y=exact_vals, mode='lines', name='Exact Solution'))
            fig.update_layout(title="ODE: Series vs Exact Solution Comparison", xaxis_title="x", yaxis_title="y")
            plot_html = pio.to_html(fig, full_html=False)

            result = {"series": series_solution, "exact": exact_solution, "avg_error": avg_error}
        except Exception as e:
            error_msg = str(e)
    return render_template('ode_compare.html', result=result, plot_html=plot_html, conclusion=conclusion,
                           error=error_msg)

if __name__ == '__main__':
    app.run(debug=True)
