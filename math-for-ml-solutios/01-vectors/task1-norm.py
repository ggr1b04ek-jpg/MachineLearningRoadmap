def norm(v, p):
    if p == float('inf'):
        return max(abs(x) for x in v)

    total = 0
    for x in v:
        sum += abs(x) ** p
    return sum ** (1 / p)

v = [float(x) for x in input("Вектор: ").split()]
p = float(input("Степень: "))

print(norm(v, p))