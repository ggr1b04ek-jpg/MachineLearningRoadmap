import numpy as np 
import matplotlib.pyplot as plt 
from scipy import stats

n = 50
p = 0.3
num_exps = 10_000

bernoulli_matrix = np.random.binomial(1, p, size=(num_exps, n))

simulated_sums = bernoulli_matrix.sum(axis=1)

k = np.arange(0, n + 1)

theoretical_pmf = stats.binom.pmf(k, n, p)

plt.figure(figsize=(10, 6))

plt.hist(simulated_sums, bins=np.arange(-0.5, n + 1.5, 1),
        density=True, alpha=0.6, color='skyblue', edgecolor='black',
        label='Симуляция (10 000 опытов)')

plt.plot(k, theoretical_pmf, 'ro-', markersize=4, linewidth=1.5,
        label=f'Теория: Binom(n={n}, p={p})')

plt.title('Эмпирическая сумма Бернулли vs Теоритическое Биноминальное', fontsize=14)
plt.xlabel('Количество успехов (k)')
plt.ylabel('Вероятность')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.show()

print(f"Теоретическое среднее (n*p): {n * p}")
print(f"Среднее из симуляции:        {simulated_sums.mean():.2f}")