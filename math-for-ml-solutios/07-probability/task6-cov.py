import numpy as np 
import matplotlib.pyplot as plt

mean = [0, 0]

true_cov = np.array([
    [2.0, 1.5],
    [1.5, 3.0]
])

num_samples = 1000

data = np.random.multivariate_normal(mean, true_cov, num_samples)

sample_cov = np.cov(data, rowvar=False)

print("Истинная ковариационная матрица:")
print(true_cov)

print("\nВосстановленная ковариационная матрица:")
print(np.round(sample_cov, 2))

print("\nСовпадают ли они (с точностью до 0.2)?", np.allclose(true_cov, sample_cov, atol=0.2))

plt.figure(figsize=(8, 8))
plt.scatter(data[:, 0], data[:, 1], alpha=0.5, edgecolor='none')

plt.title('2D Нормальное распределение с заданной ковариацией')
plt.xlabel('Признак X')
plt.ylabel('Признак Y')
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.show()