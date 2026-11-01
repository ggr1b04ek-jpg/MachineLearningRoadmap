import numpy as np
import matplotlib.pyplot as plt

def rosenbrock(v):
    x, y = v[0], v[1]
    return (1 - x)**2 + 100 * (y - x**2)**2

def grad_rosenbrock(v):
    x, y = v[0], v[1]
    dx = -2 * (1 - x) - 400 * x * (y - x**2)
    dy = 200 * (y - x**2)
    return np.array([dx, dy])

def gradient_descent_coords(f, grad_f, x0, lr=0.1, steps=50):
    x = x0.copy()
    history = [x.copy()]
    for _ in range(steps):
        x -= lr * grad_f(x)
        history.append(x.copy())
    return x, history

x0 = np.array([0.0, 1.5])
lr = 0.001
steps = 10000

x_min, hist = gradient_descent_coords(rosenbrock, grad_rosenbrock, x0, lr=lr, steps=steps)

plt.figure(figsize=(10, 8))

x_range = np.linspace(-2, 2, 100)
y_range = np.linspace(-1, 3, 100)
X, Y = np.meshgrid(x_range, y_range)
Z = (1 - X)**2 + 100 * (Y - X**2)**2
plt.contour(X, Y, Z, levels=50, cmap='viridis', alpha=0.5)

hist_array = np.array(hist)
plt.plot(hist_array[:, 0], hist_array[:, 1], 'r.-', markersize=3, label='Trajectory')
plt.plot(x0[0], x0[1], 'go', label='Start', markersize=10)
plt.plot(1, 1, 'bo', label='Minimum (1, 1)', markersize=10)

plt.title('GD на функции Розенброка (Zoomed In)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

print(f"Финальная точка: {x_min}")
if not np.any(np.isnan(x_min)):
    print(f"Значение функции: {rosenbrock(x_min)}")