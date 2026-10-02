import numpy as np

matrices = [np.random.randn(5, 5) for _ in range(10)]

matrices[2][2] = matrices[2][0] +matrices[2][1]
matrices[7][4] = 2 * matrices[7][2]

degenerate_indices = []

for i, M in enumerate(matrices):
    rank = np.linalg.matrix_rank(M)

    if rank < 5:
        degenerate_indices.append(i)

print(f"Вырожденные матрицы: {degenerate_indices}")

if degenerate_indices == [2, 7]:
    print("Все правлиьно")
else:
    print("Найдены не те")