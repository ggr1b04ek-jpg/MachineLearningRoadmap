import numpy as np
import matplotlib.pyplot as plt

def power_method(A, tol=1e-6, max_iter=1000):
    n = A.shape[0]
    if n != A.shape[1]:
        raise ValueError("Матрица должна быть квадратной")
    
    v = np.random.rand(n)
    v = v / np.linalg.norm(v)

    eigen_value_old = 0.0
    history = []

    for i in range(max_iter):
        w = A @ v
        eigen_value = np.linalg.norm(w)

        v = w / np.linalg.norm(w)
        
        error = abs(eigen_value - eigen_value_old)
        history.append(error)

        if error < tol:
            print(f"Сошлись за {i+1} итераций, лямбда: {eigen_value:.10f}")
            break
        
        eigen_value_old = eigen_value
    else:
        print(f"Не сошлись за {max_iter} итераций, последняя ошибка: {history[-1]:.2f}")

    return eigen_value, v, history

A = np.array([
    [4, 1, 0, 0],
    [1, 3, 1, 0],
    [0, 1, 2, 1],
    [0, 0, 1, 1]
], dtype=np.float64)

eigen_value, eigen_vector, history = power_method(A)

eigen_values_np, eigen_vectors_np = np.linalg.eig(A)
idx_max = np.argmax(np.abs(eigen_values_np))
lambda_true = eigen_values_np[idx_max]
vector_true = eigen_vectors_np[:, idx_max]

print(f" Сравнение результатов:")
print(f"   Степенной метод:  λ = {eigen_value:.10f}")
print(f"   NumPy (эталон):   λ = {lambda_true.real:.10f}")
print(f"   Абсолютная ошибка: {abs(eigen_value - lambda_true.real):.2e}")

sign = np.sign(np.dot(eigen_vector, vector_true.real))
vector_error = np.linalg.norm(eigen_vector - sign * vector_true.real)
print(f"   Ошибка вектора:    {vector_error:.2e}")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.semilogy(history, 'bo-', markersize=4, linewidth=1.5)
plt.xlabel('Номер итерации')
plt.ylabel('Абсолютная ошибка')
plt.title('Сходимость степенного метода')
plt.grid(True, alpha=0.3)
plt.axhline(y=1e-8, color='r', linestyle='--', label=f'tol={1e-8}')
plt.legend()

plt.subplot(1, 2, 2)
plt.imshow(A, cmap='Blues', interpolation='nearest')
plt.colorbar(label='Значение элемента')
plt.title('Исходная матрица A')
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        plt.text(j, i, f'{A[i,j]:.0f}', ha='center', va='center', fontsize=12)

plt.tight_layout()
plt.show()