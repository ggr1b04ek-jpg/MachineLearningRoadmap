import numpy as np
import matplotlib.pyplot as plt

n_samples = 1_000_000
threshold = 5.0

normal_data = np.random.normal(loc=0, scale=1, size=n_samples)
cauchy_data = np.random.standard_cauchy(size=n_samples)

normal_outliers_count = (np.abs(normal_data)  > threshold).sum()
cauchy_outliers_count = (np.abs(cauchy_data) > threshold).sum()

normal_out_ratio = normal_outliers_count / n_samples
cauchy_out_ratio = cauchy_outliers_count / n_samples

print(f"Нормальное: {normal_outliers_count:>8} раз (доля: {normal_out_ratio:.6%})")
print(f"Коши:       {cauchy_outliers_count:>8} раз (доля: {cauchy_out_ratio:.6%})")
if normal_outliers_count > 0:
    print(f"Во сколько раз Коши чаще даёт выбросы: {cauchy_outliers_count / normal_outliers_count:.0f} раз")
else:
    print("Нормальное распределение не дало ни одного выброса (0 раз).")

limit = 10
normal_clipped = normal_data[(normal_data >= -limit) & (normal_data <= limit)]
cauchy_clipped = cauchy_data[(cauchy_data >= -limit) & (cauchy_data <= limit)]

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(normal_clipped, bins=100, density=True, color='skyblue', alpha=0.7)
plt.title('Нормальное распределение (обрезано до [-10, 10])')
plt.xlabel('x')
plt.ylabel('Плотность')

plt.subplot(1, 2, 2)
plt.hist(cauchy_clipped, bins=100, density=True, color='salmon', alpha=0.7)
plt.title('Распределение Коши (обрезано до [-10, 10])')
plt.xlabel('x')
plt.ylabel('Плотность')

plt.tight_layout()
plt.show()