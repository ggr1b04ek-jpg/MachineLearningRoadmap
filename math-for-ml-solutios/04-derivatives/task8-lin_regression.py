import numpy as np

n_samples = 100
n_features = 3

X = np.random.rand(n_samples, n_features) * 10
w_true = np.array([1.5, -2.0, 0.5])
y = X @ w_true + np.random.rand(n_samples) * 2

def mse_loss(w, X, y):
    predictions = X @ w
    return np.mean((y - predictions)**2)

def mse_gradient(w, X, y):
    predictions = X @ w
    error = predictions - y
    return (2 / len(y)) * (X.T @ error)

def gradient_descent(X, y, lr=0.01, steps=1000):
    w = np.zeros(X.shape[1])
    for _ in range(steps):
        grad = mse_gradient(w, X, y)
        w -= grad * lr
    return w

lr = 0.01
steps = 2000
w_found = gradient_descent(X, y, lr=lr, steps=steps)
w_ana = np.linalg.solve(X.T @ X, X.T @ y)

print(f"Истинные веса:      {w_true}")
print(f"Найденные (GD):     {w_found}")
print(f"Аналитические:      {w_ana}")
print(f"Разница GD и Анал:  {np.abs(w_found - w_ana)}")