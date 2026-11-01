import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img = Image.open('C:/Users/user/MachineLearningRoadmap/math-for-ml-solutios/03-eigen-svd/eagle.png').convert('L')
img = img.resize((1000, 1000))
A = np.array(img, dtype=np.float64)

U, s, Vt = np.linalg.svd(A, full_matrices=False)

k_values = [5, 10, 50, 100, 500]
errors = []
sizes = []
reconstructed_img = []

max_rank = min(A.shape)

for k in k_values:
    sigma_k = np.zeros_like(s)
    sigma_k[:k] = s[:k]

    A_k = U @ np.diag(sigma_k) @ Vt
    A_k = np.clip(A_k, 0, 255)

    error = np.linalg.norm(A - A_k, 'fro')
    errors.append(error)
    
    compressed_size = k * (A.shape[0] + 1 + A.shape[1])
    sizes.append(compressed_size)

    reconstructed_img.append(A_k)

fig, axes = plt.subplots(3, 3, figsize=(18, 18))

axes[0, 0].plot(k_values, errors, 'ro-', linewidth=2, markersize=8)
axes[0, 0].set_title('Ошибка восстановления', fontsize=14)
axes[0, 0].set_xlabel('Ранг аппроксимации (k)')
axes[0, 0].set_ylabel('Ошибка')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(k_values, sizes, 'bs-', linewidth=2, markersize=8)
axes[0, 1].axhline(y=A.size, color='gray', linestyle='--', label='Оригинал')
axes[0, 1].set_title('Размер сжатых данных', fontsize=14)
axes[0, 1].set_xlabel('Ранг аппроксимации (k)')
axes[0, 1].set_ylabel('Количество чисел')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

image_positions = [(0, 2), (1, 0), (1, 1), (1, 2), (2, 0)]
titles = [f'k={k}' for k in k_values]

for i, (img, title) in enumerate(zip(reconstructed_img, titles)):
    row = (i + 2) // 3
    col = (i + 2) % 3
    axes[row, col].imshow(img, cmap='gray')
    axes[row, col].set_title(title, fontsize=14)
    axes[row, col].axis('off')

axes[2, 1].axis('off')
axes[2, 2].axis('off')

plt.suptitle('SVD Сжатие изображения 1000x1000', fontsize=20, y=1.02)
plt.tight_layout()
plt.show()

fig2, axes2 = plt.subplots(1, 5, figsize=(20, 4))
for ax, rec_img, k in zip(axes2, reconstructed_img, k_values):
    ax.imshow(rec_img, cmap='gray')
    ax.set_title(f'k = {k}', fontsize=16)
    ax.axis('off')
plt.suptitle('Визуальное качество восстановления при разных k', fontsize=18)
plt.tight_layout()
plt.show()