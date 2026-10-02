import numpy as np

def my_gauss_solver(A, b):
    n = len(b)
    A = A.astype(float).copy()
    b = b.astype(float).copy()

    for i in range(n):
        max_row = np.argmax(np.abs(A[i:, i])) + i
        if abs(A[max_row, i]) < 1e-10:
            raise ValueError("Матрица вырождена или система не имеет единственного решения")
        
        if max_row != i:
            A[[i, max_row]] = A[[max_row, i]]
            b[[i, max_row]] = b[[max_row, i]]
        for j in range(i + 1, n):
            factor = A[j, i] / A[i, i]
            A[j, i:] -= factor * A[i, i:]
            b[j] -= factor * b[i]
    
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        sum_ax = np.dot(A[i, i+1:], x[i+1:])
        x[i] = (b[i] - sum_ax) / A[i, i]
    
    return x

n = 5
A = np.random.randn(n, n)
true_x = np.array([1, 2, 3, 4, 5])
b = A @ true_x

x_my = my_gauss_solver(A, b)
x_np = np.linalg.solve(A, b)

print(f"Совпадение: {np.allclose(x_my, x_np)}")
print(f"Максимальная ошибка: {np.max(np.abs(x_my - x_np))}")