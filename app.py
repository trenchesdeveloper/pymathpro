# app.py
from flask import Flask, render_template, request
import logging
from modules import symbolic, numerical  # Import both modules

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
                x_val = float(request.form.get('x_val'))
                h = float(request.form.get('h') or 1e-5)
                result = numerical.numerical_derivative(func_str, x_val, h)
            elif method == 'integrate':
                a = float(request.form.get('a'))
                b = float(request.form.get('b'))
                n = int(request.form.get('n'))
                result = numerical.trapezoidal_integration(func_str, a, b, n)
            elif method == 'rk4':
                x0 = float(request.form.get('x0'))
                y0 = float(request.form.get('y0'))
                h = float(request.form.get('h'))
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
                x0 = float(request.form.get('x0'))
                result = numerical.newton_raphson(func_str, x0)
            else:
                error = "Invalid numerical method selected."
        except Exception as e:
            error = str(e)
    return render_template('numerical.html', result=result, error=error, plot_html=plot_html, xs=xs, ys=ys)

if __name__ == '__main__':
    app.run(debug=True)
