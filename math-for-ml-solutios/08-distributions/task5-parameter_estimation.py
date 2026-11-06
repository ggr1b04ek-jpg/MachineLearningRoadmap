import numpy as np
from scipy.optimize import minimize

def neg_log_likelihood(params, x):
  mu = params[0]
  sigma = params[1]
  if sigma <= 0:
        return np.inf
  n = len(x)
  neg_ll = n / 2 * np.log(2 * np.pi) + n * np.log(sigma) + 1 / (2 * sigma**2) * np.sum((x - mu)**2)
  return neg_ll

mu_real = 3
sigma_real = 2
n = 1000
x = np.random.normal(loc=mu_real, scale=sigma_real, size=n)

mu_mm = x.mean()
sigma_mm = x.std(ddof=1)

initial_guess = [x.mean(), x.std()]

result = minimize(
    neg_log_likelihood,
    initial_guess,
    args=(x,),
    method='Nelder-Mead'
)

mu_mle = result.x[0]
sigma_mle = result.x[1]

print(f"{'Параметр':<14} | {'Истинное':<10} | {'Метод моментов':<15} | {'MLE (Оптимизация)':<15}")
print(f"Среднее (μ)    | {mu_real:<10.4f} | {mu_mm:<15.4f} | {mu_mle:<15.4f}")
print(f"Ст. откл. (σ)  | {sigma_real:<10.4f} | {sigma_mm:<15.4f} | {sigma_mle:<15.4f}")