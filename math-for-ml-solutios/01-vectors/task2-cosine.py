from math import sqrt
import numpy as np
import random

def norm(v, p):
    if p == float('inf'):
        return max(abs(x) for x in v)

    total = 0
    for x in v:
        total += abs(x) ** p
    return total ** (1 / p)

def cosine(a, b):
    if len(a) != len(b):
        raise ValueError("Векторы должны иметь одинаковые размерности")
    
    norm_a = norm(a, 2)
    norm_b = norm(b, 2)

    if norm_a == 0 or norm_b == 0:
        raise ValueError("Нельзя вычислить косинус для нулевого вектора")

    return sum(a[i] * b[i] for i in range(len(a))) / (norm_a * norm_b)
        

print("Сравнение реализации с NumPy:")
for i in range(10):
    a_np = np.random.rand(5)
    b_np = np.random.rand(5)

    my_cosine = cosine(a_np.tolist(), b_np.tolist())
    np_cosine = np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np))

    print(f"{i+1}: Мой: {my_cosine}  Numpy: {np_cosine}  Разница: {abs(my_cosine - np_cosine):.10f}")