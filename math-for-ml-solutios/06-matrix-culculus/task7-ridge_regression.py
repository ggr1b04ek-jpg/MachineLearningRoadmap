import numpy as np 

n_samples = 10
n_features = 20
X = np.random.randn(n_samples, n_features)
y = np.random.randn(n_samples)

try:
    w_ols = np.linalg.solve(X.T @ X, X.T @ y)
    print("OLS решил:", w_ols)
except np.linalg.LinAlgError as e:
    print("OLS упал:", e)

print("det(X.T @ X) = ", np.linalg.det(X.T @ X))

def ridge_regression(X, y, lam):
    n_features = X.shape[1]
    I = np.eye(n_features)
    A = X.T @ X + lam * I
    b = X.T @ y
    return np.linalg.solve(A, b)

for lam in [0.001, 0.01, 0.1, 1.0]:
    w_ridge = ridge_regression(X, y, lam)
    print(f"lam={lam}: ||w|| = {np.linalg.norm(w_ridge):.4f}")

import matplotlib.pyplot as plt

lambdas = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
norms = []

for lam in lambdas:
    w = ridge_regression(X, y, lam)
    norms.append(np.linalg.norm(w))

plt.figure(figsize=(10, 6))
plt.semilogx(lambdas, norms, 'o-', linewidth=2, markersize=8)
plt.xlabel('λ (lambda)', fontsize=12)
plt.ylabel('||w||² (норма весов)', fontsize=12)
plt.title('Эффект регуляризации: как λ сжимает веса', fontsize=14)
plt.grid(True, alpha=0.3)
plt.axhline(y=np.linalg.norm(w_ols), color='r', linestyle='--', label='OLS (без регуляризации)')
plt.legend()
plt.show()

print(f"Норма весов OLS: {np.linalg.norm(w_ols):.4f}")
print(f"Норма весов Ridge (λ=1.0): {norms[-1]:.4f}")
print(f"Сжатие весов: {np.linalg.norm(w_ols) / norms[-1]:.2f}x")