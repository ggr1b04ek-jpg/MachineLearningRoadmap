import numpy as np 
import matplotlib.pyplot as plt 
from scipy import stats

n_vals = [1, 2, 5, 10, 30]

num_exps = 10_000

fig, axes = plt.subplots(1, len(n_vals), figsize=(15, 3))

for i, n in enumerate(n_vals):
    uniform_matrix = np.random.uniform(0, 1, size=(num_exps, n))
    mn = uniform_matrix.mean(axis=1)
    axes[i].hist(mn, bins=40, density=True, alpha=0.7, color='skyblue', edgecolor='black')

    axes[i].set_title(f'n={n}')
    axes[i].set_ylim(0, 2.5)
    if i == 0:
        axes[i].set_ylabel('Плотность')

plt.suptitle('ЦПТ: от равномерного к нормальному', fontsize=14)
plt.tight_layout()
plt.show()