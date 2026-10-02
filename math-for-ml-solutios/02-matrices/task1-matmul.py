import numpy as np
import time

def my_matmul(A, B):
    m, n1 = A.shape
    n2, p = B.shape

    if n1 != n2:
        raise AssertionError("Внутренние размеры матриц должны совпадать")
    
    C = np.zeros((m, p))

    for i in range(m):
        for j in range(p):
            for k in range(n1):
                C[i, j] += A[i, k] * B[k, j]
    
    return C

np.random.seed(42)
size = 200
A = np.random.randn(size, size)
B = np.random.randn(size, size)

my_result = my_matmul(A, B)
np_result = A @ B

print("Результаты совпадают:", np.allclose(my_result, np_result))
print("Максимальная разница:", np.max(np.abs(my_result - np_result)))

start = time.time()
for _ in range(3):
    my_matmul(A, B)
my_time = (time.time() - start) / 3

start = time.time()
for _ in range(3):
    A @ B
np_time = (time.time() - start) / 3

print(f"Моё время: {my_time}, NumPy время: {np_time}, NumPy быстрее в {my_time / np_time:.1f} раз")