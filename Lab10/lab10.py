import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

# Funcție pentru generarea datelor
def generate_data(order, num_points=500):
    x = np.linspace(-3, 3, num_points)
    y = 5 * x**2 - 2 * x + 1 + np.random.normal(0, 0.5, size=num_points)
    return x, y

# Funcție pentru modelul polinomial
def run_polynomial_model(x, y, order, sd_values):
    x_p = np.vstack([x**i for i in range(1, order+1)])
    x_s = (x_p - x_p.mean(axis=1, keepdims=True)) / x_p.std(axis=1, keepdims=True)
    y_s = (y - y.mean()) / y.std()

    with pm.Model() as model_poly:
        α = pm.Normal('α', mu=0, sigma=1)
        β = pm.Normal('β', mu=0, sigma=sd_values, shape=order)
        ε = pm.HalfNormal('ε', 5)
        μ = α + pm.math.dot(β, x_s)
        y_pred = pm.Normal('y_pred', mu=μ, sigma=ε, observed=y_s)
        idata_poly = pm.sample(2000, return_inferencedata=True)

    α_post = idata_poly.posterior['α'].mean(("chain", "draw")).values
    β_post = idata_poly.posterior['β'].mean(("chain", "draw")).values
    y_post = α_post + np.dot(β_post, x_s)

    plt.plot(x_s[0], y_post, label=f'model order {order}')

# 1. a)
order = 5
x, y = generate_data(order)
plt.scatter(x, y, c='C0', marker='.')

run_polynomial_model(x, y, order, sd_values=10)

# 1. b)
plt.show()

run_polynomial_model(x, y, order, sd_values=100)

run_polynomial_model(x, y, order, sd_values=np.array([10, 0.1, 0.1, 0.1, 0.1]))

# 2
x_500, y_500 = generate_data(order, num_points=500)
plt.scatter(x_500, y_500, c='C0', marker='.')

run_polynomial_model(x_500, y_500, order, sd_values=10)

# 3
order_cubic = 3
x_cubic, y_cubic = generate_data(order_cubic, num_points=500)

plt.scatter(x_cubic, y_cubic, c='C0', marker='.')

with pm.Model() as model_cubic:
    α = pm.Normal('α', mu=0, sigma=1)
    β = pm.Normal('β', mu=0, sigma=10, shape=3)
    ε = pm.HalfNormal('ε', 5)
    μ = α + pm.math.dot(β, x_cubic[:3])
    y_pred_cubic = pm.Normal('y_pred', mu=μ, sigma=ε, observed=y_cubic[:3])
    idata_cubic = pm.sample(2000, return_inferencedata=True)

α_cubic_post = idata_cubic.posterior['α'].mean(("chain", "draw")).values
β_cubic_post = idata_cubic.posterior['β'].mean(("chain", "draw")).values
y_cubic_post = α_cubic_post + np.dot(β_cubic_post, x_cubic)

plt.plot(x_cubic, y_cubic_post, 'C4', label='model cubic')

plt.legend()
plt.show()
