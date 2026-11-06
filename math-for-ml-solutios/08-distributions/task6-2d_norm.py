import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

mu = [0, 0]
cov = [[1, 0.9],
       [0.9, 1]]
n_samples = 1000

data = np.random.multivariate_normal(mu, cov, n_samples)

x = np.linspace(-4, 4, 100)
y = np.linspace(-4, 4, 100)
X, Y = np.meshgrid(x, y)

pos = np.dstack((X, Y))

Z = stats.multivariate_normal.pdf(pos, mu, cov)

plt.figure(figsize=(8, 8))

plt.scatter(data[:, 0], data[:, 1], alpha=0.5, color='skyblue', edgecolors='none', label='Данные')

contour = plt.contour(X, Y, Z, levels=5, colors='red', linewidths=1.5)

plt.title('2D Нормальное распределение с сильной положительной корреляцией', fontsize=14)
plt.xlabel('Признак X')
plt.ylabel('Признак Y')
plt.axhline(0, color='black', linewidth=1, alpha=0.5)
plt.axvline(0, color='black', linewidth=1, alpha=0.5)
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.show()