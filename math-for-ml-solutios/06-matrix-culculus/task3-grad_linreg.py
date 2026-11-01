import numpy as np 

def grad_linreg(X, y, w):
    return 2 * X.T @ (X @ w - y)

def gd(X, y, grad_func, lr=0.01, iters=1000):
    w = np.random.randn(X.shape[1])

    for it in range(iters):
        w -= lr * grad_func(X, y, w)
    
    return w

X = np.random.randn(100, 5)
w_true = np.random.randn(5)
y = X @ w_true + np.random.uniform(-1.0, 1.0, 100)

w_pred = gd(X, y, grad_linreg, lr=0.005, iters=10000)

A = X.T @ X
b = X.T @ y
w_np_pred = np.linalg.solve(A, b)

print(np.allclose(w_pred, w_np_pred))