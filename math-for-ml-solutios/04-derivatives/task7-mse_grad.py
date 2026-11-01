import numpy as np

X = np.random.rand(100, 1) * 10
y = 3 * X.flatten() + np.random.randn(100) * 2

def mse_loss(w, X, y):
    predictions = X @ w
    return np.mean((y - predictions)**2)

def mse_gradient_analytical(w, X, y):
    predictions = X @ w
    error = predictions - y
    grad = (2 / len(y)) * (X.T @ error)
    return grad.item()

def mse_gradient_numerical(w, X, y, h=1e-5):
    return (mse_loss(w + h, X, y) - mse_loss(w - h, X, y)) / (2 * h)

w_test = np.array([1.0])
grad_num = mse_gradient_numerical(w_test, X, y)
grad_ana = mse_gradient_analytical(w_test, X, y)

print(f"Численный градиент: {grad_num:.6f}")
print(f"Аналитический градиент: {grad_ana:.6f}")
print(f"Разница: {abs(grad_num - grad_ana):.2e}")