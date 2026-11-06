import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

n_samples = 100
x = np.linspace(0, 10, n_samples)

true_w, true_b, true_sigma = 2.0, 1.0, 1.0
noise = np.random.normal(0, true_sigma, n_samples)
y = true_w * x + true_b + noise

def mse_loss(params, x, y):
  w, b = params[0], params[1]
  y_pred = w * x + b
  return ((y - y_pred)**2).mean()

def nll_gaussian(params, x, y, sigma):
  w, b = params[0], params[1]
  y_pred = w * x + b
  n = len(y)
  nll = (n / 2) * np.log(2 * np.pi * sigma**2) + (1 / (2 * sigma**2)) * np.sum((y - y_pred)**2)
  return nll

initial_guess = [0.0, 0.0]

result_mse = minimize(mse_loss, initial_guess, args=(x, y), method='Nelder-Mead')
w_mse, b_mse = result_mse.x

result_nll = minimize(nll_gaussian, initial_guess, args=(x, y, true_sigma), method='Nelder-Mead')
w_nll, b_nll = result_nll.x

print(f"Истинные параметры:    w = {true_w:.4f}, b = {true_b:.4f}")
print(f"Найдено через MSE:     w = {w_mse:.4f}, b = {b_mse:.4f}")
print(f"Найдено через NLL:     w = {w_nll:.4f}, b = {b_nll:.4f}")

plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.5, label='Данные с шумом', color='gray')
plt.plot(x, true_w * x + true_b, 'k--', label='Истинная прямая', linewidth=2)
plt.plot(x, w_mse * x + b_mse, 'r-', label=f'MSE (w={w_mse:.2f})', linewidth=2)
plt.plot(x, w_nll * x + b_nll, 'g:', label=f'NLL (w={w_nll:.2f})', linewidth=3)
plt.legend()
plt.title('MSE и Gaussian MLE дают одинаковый результат')
plt.grid(True, alpha=0.3)
plt.show()