import numpy as np

n = 1000
true_lam = 4

x = np.random.poisson(lam=true_lam, size=n)

lam_hat = x.mean()

print(f"Истинная интенсивность (λ):   {true_lam:.4f}")
print(f"Количество наблюдений (n):    {n}")
print(f"Суммарное количество событий: {np.sum(x)}")
print(f"MLE-оценка интенсивности (λ̂): {lam_hat:.4f}")