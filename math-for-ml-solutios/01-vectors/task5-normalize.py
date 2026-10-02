def normalize(v):
    norm_v = sum(x ** 2 for x in v) ** 0.5

    if norm_v == 0:
        raise ValueError("Нельзя нормализовать нулевой вектор")

    return [x / norm_v for x in v]

v = [float(x) for x in input("Вектор v: ").split()]

print(normalize(v))