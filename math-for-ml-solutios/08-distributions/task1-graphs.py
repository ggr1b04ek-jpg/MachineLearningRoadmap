import numpy as np 
import matplotlib.pyplot as plt 
from scipy import stats 

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

'''Бернулли'''
p_values = [0.2, 0.5, 0.8]
x_bern = [0, 1]

for p in p_values:
    pmf = stats.bernoulli.pmf(x_bern, p)
    axes[0].stem(x_bern, pmf, linefmt=f'-', markerfmt='o', basefmt='-', label=f'p={p}')

axes[0].set_title('Бернулли (PMF)')
axes[0].set_xlabel('X')
axes[0].set_ylabel('P(X=x)')
axes[0].legend()
axes[0].set_ylim(0, 1.1)

'''Биноминальное'''
n_vals = [5, 10]
p_vals_binom = [0.3, 0.7]
x_binom = np.arange(0, 60)

for n in n_vals:
    for p in p_vals_binom:
        pmf = stats.binom.pmf(x_binom, n, p)
        axes[1].plot(x_binom, pmf, label=f'n={n}, p={p}', alpha=0.7)

axes[1].set_title('Биноминальное (PMF)')
axes[1].set_xlabel('k (число успехов)')
axes[1].set_ylabel('P(X=k)')
axes[1].legend()

'''Нормальное'''
x_norm = np.linspace(-6, 6, 400)
params_norm = [(0, 1), (0, 2), (-2, 0.5)]

for mu, sigma in params_norm:
    pdf = stats.norm.pdf(x_norm, loc=mu, scale=sigma)
    axes[2].plot(x_norm, pdf, label=f'mu={mu}, sigma={sigma}', linewidth=2)

axes[2].set_title('Нормальное (PDF)')
axes[2].set_xlabel('x')
axes[2].set_ylabel('P(x)')
axes[2].legend()

'''Пуассоновское'''
lambda_vals = [2, 5, 10]
x_poisson = np.arange(0, 25)

for lam in lambda_vals:
    pmf = stats.poisson.pmf(x_poisson, lam)
    axes[3].stem(x_poisson, pmf, linefmt=f'-', markerfmt='o', basefmt='-',label=f'lam={lam}')

axes[3].set_title('Пуассоновское (PMF)')
axes[3].set_xlabel('k (число событий)')
axes[3].set_ylabel('P(X=k)')
axes[3].legend()

'''Экспоненциальное'''
x_exp = np.linspace(0, 5, 400)
lambda_exp_vals = [0.5, 1.0, 2.0]
for lam in lambda_exp_vals:
    pdf = stats.expon.pdf(x_exp, scale=1/lam)
    axes[4].plot(x_exp, pdf, label=f'lam={lam}', linewidth=2)

axes[4].set_title('Экспоненциальное (PDF)')
axes[4].set_xlabel('x (время)')
axes[4].set_ylabel('p(x)')
axes[4].legend()

axes[5].axis('off')

plt.suptitle('Ключевые распределения: PMF vs PDF', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()