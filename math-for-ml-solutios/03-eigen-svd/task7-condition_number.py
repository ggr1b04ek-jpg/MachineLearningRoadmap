import numpy as np
import matplotlib.pyplot as plt

matrices = {
    "Хорошая":       np.array([[1, 0], [0, 1]]),
    "Средняя":     np.array([[1, 0], [0, 1e-6]]),
    "Плохая":     np.array([[1, 0], [0, 1e-12]])
}

results = []

noise = 1e-14
X_true = np.array([1.0, 1.0])

for name, A in matrices.items():
    s = np.linalg.svd(A, compute_uv=False)
    cond_nums = s[0] / s[-1]
    
    b = A @ X_true
    b_noisy = b + np.random.normal(0, noise, size=b.shape)

    X_sol = np.linalg.solve(A, b_noisy)

    error = np.linalg.norm(X_sol - X_true) / np.linalg.norm(X_true)

    results.append((cond_nums, error))
    print(f"{name:<20} | {cond_nums:.2e}     | {error:.2e}")

cond_nums, errors = zip(*results)

plt.figure(figsize=(10, 6))
plt.loglog(cond_nums, errors, 'ro-', linewidth=2, markersize=10)
plt.xlabel('Число обусловленности', fontsize=12)
plt.ylabel('Относительная ошибка решения', fontsize=12)
plt.title('Рост ошибки при ухудшении обусловленности', fontsize=14)
plt.grid(True, which="both", ls="-", alpha=0.5)

plt.plot(cond_nums, [c * noise for c in cond_nums], 
         'b--', label=f'Теория: ε · κ (ε={noise:.0e})')
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()