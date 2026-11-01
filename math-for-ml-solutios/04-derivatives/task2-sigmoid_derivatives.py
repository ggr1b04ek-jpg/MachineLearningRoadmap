import numpy as np

"""
sigmoid = f = 1 / (1 + e^(-x))
df = -1 * (-e^x) / (1 + e^(-x))^2 = e^x / (1 + e^(-x))^2
1 - f = e^(-x) / (1 + e^(-x))
f * (1 - f) = df
"""

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative_analytical(x):
    s = 1 / (1 + np.exp(-x))
    return s * (1 - s)

def sigmoid_derivative_numerical(x, h=1e-5):
    return (sigmoid(x + h) - sigmoid(x - h)) / (2 * h)

x_values = np.linspace(-5, 5, 100)
errors = []

for x in x_values:
    ds_an = sigmoid_derivative_analytical(x)
    ds_num = sigmoid_derivative_numerical(x)
    errors.append(np.abs(ds_an - ds_num))

print(f"MAE между аналитической и численной производными: {np.mean(errors)}")