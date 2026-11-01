import numpy as np 

lam = 3.0
size = 10_000

data = np.random.poisson(lam, size)

sample_mean = data.mean()
sample_var = data.var()

print(f"Теоретическое λ (и среднее, и дисперсия): {lam}")
print(f"Выборочное среднее (E[X]): {sample_mean:.4f}")
print(f"Выборочная дисперсия (Var[X]): {sample_var:.4f}")
print(f"Разница со средним: {abs(sample_mean - lam):.4f}")
print(f"Разница с дисперсией: {abs(sample_var - lam):.4f}")