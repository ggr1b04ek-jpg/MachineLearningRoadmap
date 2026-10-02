import numpy as np
import time

def my_matmul_vectorized(A, B):
    A_expanded = A[:, :, np.newaxis]
    B_expanded = B[np.newaxis, :, :]

    products = A_expanded * B_expanded

    C = np.sum(products, axis=1)

    return C

np.random.seed(42)
size = 200
A = np.random.randn(size, size)
B = np.random.randn(size, size)

my_result = my_matmul_vectorized(A, B)
np_result = A @ B

print("Результаты совпадают:", np.allclose(my_result, np_result))
print("Максимальная разница:", np.max(np.abs(my_result - np_result)))

start = time.time()
for _ in range(10):
    my_matmul_vectorized(A, B)
vec_time = (time.time() - start) / 10

start = time.time()
for _ in range(100):
    A @ B
np_time = (time.time() - start) / 100

print(f"Векторизованное matmul: {vec_time:.4f} сек")
print(f"NumPy: {np_time:.6f} сек")
print(f"NumPy быстрее в {vec_time / np_time:.1f} раз")