import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

p_values = np.linspace(0, 1, 500)

alpha_prior = 2
beta_prior = 2

prior_pdf = stats.beta.pdf(p_values, alpha_prior, beta_prior)

k_successes = 4
n_failures = 1

alpha_post = alpha_prior + k_successes
beta_post = beta_prior + n_failures

post_pdf = stats.beta.pdf(p_values, alpha_post, beta_post)

plt.figure(figsize=(10, 6))

plt.plot(p_values, prior_pdf, label=f'Prior: Beta({alpha_prior}, {beta_prior})', linewidth=2, color='gray', linestyle='--')

plt.plot(p_values, post_pdf, label=f'Posterior: Beta({alpha_post}, {beta_post})', linewidth=3, color='blue', linestyle='-')

plt.axvline(x=0.8, color='red', linestyle=':', linewidth=2, label='Частотная оценка (0.8)')

plt.title('Построение Posterior с Beta(2, 2)')
plt.xlabel('Вероятность')
plt.ylabel('Плотность вероятности')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, 1)

plt.show()

mode_prior = (alpha_prior - 1) / (alpha_prior + beta_prior - 2)
mode_post = (alpha_post - 1) / (alpha_post + beta_post - 2)

print(f"Мода Prior: {mode_prior:.2f}")
print(f"Мода Posterior: {mode_post:.2f}")
print(f"Частотная оценка: 0.80")