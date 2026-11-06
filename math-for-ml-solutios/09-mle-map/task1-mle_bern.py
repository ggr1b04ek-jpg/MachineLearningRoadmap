import numpy as np

n = 100
true_p = 0.3

x = np.random.binomial(n=1, p=true_p, size=n)

p_hat = x.mean()

print(f"Истинная вероятность (p):   {true_p:.1f}")
print(f"Количество бросков (n):     {n}")
print(f"Количество успехов (S):     {np.sum(x)}")
print(f"MLE-оценка вероятности (p̂): {p_hat:.2f}")
print(f"Проверка формулы (sum(x)/n): {np.sum(x) / n:.2f}")