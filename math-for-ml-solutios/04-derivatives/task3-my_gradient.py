import numpy as np

def grad(f, x):
    grad_vector = np.zeros_like(x)
    h = 1e-5

    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()

        x_plus[i] += h
        x_minus[i] -= h

        df = (f(x_plus) - f(x_minus)) / (2 * h)
        grad_vector[i] = df

    return grad_vector

f = lambda v: np.sum(v**2)
x = np.array([1.0, 2.0, 3.0])

my_grad = grad(f, x)
expected = 2 * x

print(f"Ожидаемый градиент: {expected}, мой: {my_grad}, разница: {np.abs(my_grad - expected)}")