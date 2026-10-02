import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = fetch_california_housing()
X = data.data
y = data.target

ones = np.ones((X.shape[0], 1))
X_with_bias = np.hstack([ones, X])

XTX = X_with_bias.T @ X_with_bias
XTy = X_with_bias.T @ y
w_hat = np.linalg.solve(XTX, XTy)

print(f"Найденные веса (первые 5): {w_hat[:5]}")

y_pred_manual = X_with_bias @ w_hat
mse_manual = mean_squared_error(y, y_pred_manual)

model_sklearn = LinearRegression()
model_sklearn.fit(X, y)
y_pred_sklearn = model_sklearn.predict(X)
mse_sklearn = mean_squared_error(y, y_pred_sklearn)

print(f"MSE manual {mse_manual:.4f}")
print(f"MSE sklearn {mse_sklearn:.4f}")
print(f" Разница в MSE {abs(mse_manual - mse_sklearn):.10f}")