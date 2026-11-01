import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(f, grad_f, x0, lr=0.1, steps=50):
    x = x0.copy()
    history = [f(x)]
    for _ in range(steps):
        x -= lr * grad_f(x)
        history.append(f(x))
    return x, history

lr_values = [0.001, 0.1, 0.5, 1.0]

f = lambda v: v[0]**2 + 3*v[1]**2
grad_f = lambda v: np.array([2*v[0], 6*v[1]])
x0 = np.array([3.0, 4.0])

plt.figure(figsize=(10, 6))

for lr in lr_values:
    _, hist = gradient_descent(f, grad_f, x0, lr=lr, steps=50)
    plt.plot(hist, label=f'lr = {lr}')

plt.title('Сходимость градиентного спуска при разных learning rates')
plt.xlabel('Шаг (Step)')
plt.ylabel('Значение функции (Loss)')
plt.legend()
plt.grid(True)
plt.yscale('log')
plt.show()