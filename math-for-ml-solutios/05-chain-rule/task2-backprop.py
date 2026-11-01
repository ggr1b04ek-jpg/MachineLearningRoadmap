import numpy as np

x = 2.0
y = 1.0
w = 0.5
b = 0.1
lr = 0.1
steps = 1000

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

print(f"Начальное предсказание: {sigmoid(w*x + b):.4f}")

for i in range(steps):
    z = w * x + b
    a = sigmoid(z)
    l = (a - y)**2
    
    dl_da = 2 * (l - a)
    da_dz = a * (1 - a)
    dz_dw = x
    dz_db = 1.0

    dl_dw = dl_da * da_dz * dz_dw
    dl_db = dl_da * da_dz * dz_db

    w -= lr * dl_dw
    b -= lr * dl_db
    
    if (i + 1) % 200 == 0:
        print(f"Шаг {i+1}: Loss = {l:.6f}, Предсказание: {a:.4f}, w = {w:.4f}, b = {b:.4f}")

print(f"Итоговое предсказание: {sigmoid(w * x + b):.6f} (цель:{y:.6f})")