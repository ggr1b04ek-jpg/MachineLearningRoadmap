import numpy as np 
import matplotlib.pyplot as plt 
from scipy import stats

n = 1000

norm = np.random.normal(loc=0, scale=1, size=n)
exp = np.random.exponential(scale=1, size=n)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

stats.probplot(norm, dist="norm", plot=axes[0])
axes[0].set_title("Нормальные данные vs Нормальное распределение")
axes[0].grid(True, alpha=0.3)

stats.probplot(exp, dist="norm", plot=axes[1])
axes[1].set_title("Экспоненциальные данные vs Нормальное распределение")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()