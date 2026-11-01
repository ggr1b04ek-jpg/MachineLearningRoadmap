import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = X**2 + 3*Y**2

plt.figure(figsize=(8, 8))
plt.contour(X, Y, Z, levels=20, cmap='viridis')

points = np.array([
    [2.0, 0.0],
    [-2.0, 1.0],
    [1.0, -2.0],
    [-1.5, -1.5],
    [0.5, 2.5]
])

for p in points:
    px, py = p
    dx = -2 * px
    dy = -6 * py

    plt.quiver(px, py, dx, dy, color='red', angles='xy', scale_units='xy', scale=1)
    plt.plot(px, py, 'ko')

plt.title('Линия уровня f(x,y) = x^2 + 3*y^2 и векторы антиградиента')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.axis('equal')
plt.show()