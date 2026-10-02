import random

v = [random.randint(0,100) for i in range(100)]
u = [0] * 100

for i in range(99):
    u[i] = random.randint(-10,10)

last_index = 99
if v[last_index] == 0:
    for i in range(100):
        if v[i] != 0:
            last_index = i
            break

partial_sum = 0
for i in range(99):
    partial_sum += v[i] * u[i]

u[last_index] = -partial_sum / v[last_index]

dot_product = sum(x * y for x, y in zip(v, u))
print(f"Скалярное произведение: {dot_product}")
print(f"Вектор u: {u}")