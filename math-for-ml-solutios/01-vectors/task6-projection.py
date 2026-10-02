import matplotlib.pyplot as plt

def proj(a, b):
    prod_a_b = sum(x * y for x, y in zip(a, b))
    prod_b_b = sum(x * x for x in b)

    if prod_b_b == 0:
        raise ValueError("Нельзя вычислить проекцию на нулевой вектор")

    coef = prod_a_b / prod_b_b

    return [x * coef for x in b]

a = [float(x) for x in input("Вектор a: ").split()]
b = [float(x) for x in input("Вектор b: ").split()]
projection = proj(a, b)
print(projection)

fig, ax = plt.subplots(figsize=(8,8))

ax.quiver(0, 0, a[0], a[1], angles='xy', scale_units='xy', scale=1, color='blue', label='Вектор a')
ax.quiver(0, 0, b[0], b[1], angles='xy', scale_units='xy', scale=1, color='red', label='Вектор b')
ax.quiver(0, 0, projection[0], projection[1], angles='xy', scale_units='xy', scale=1, color='green', label='Проекция')
ax.plot([a[0], projection[0]],[a[1], projection[1]], 'k--', alpha=0.5)

ax.set_xlim(-1, max(a[0], b[0]) + 1)
ax.set_ylim(-1, max(a[1], b[1]) + 1)
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)
ax.grid(color='gray', linestyle=':', linewidth=0.5)
ax.legend()
ax.set_aspect('equal')

plt.title("Проекция вектора a на вектор b")
plt.show()