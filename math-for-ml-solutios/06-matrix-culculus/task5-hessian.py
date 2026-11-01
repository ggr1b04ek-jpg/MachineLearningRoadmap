import numpy as np 
import matplotlib.pyplot as plt 

# f(x, y) = x² + 3y² - 2xy
#
# Первые производные:
# ∂f/∂x = 2 * x - 2 * y
# ∂f/∂y = 6 * y - 2 * x
#
# Вторые производные:
# ∂²f/∂x² = 2
# ∂²f/∂y² = 6
# ∂²f/∂x∂y = -2
#
# Гессиан H = [[2 , -2],
#              [-2 , 6]]

H = np.array([
    [2, -2],
    [-2, 6]
])

print(f"Гессиан:\n{H}")

eigenvalues = np.linalg.eigvals(H)
print(f"Собственные значения: {eigenvalues}")
if np.all(eigenvalues > 0):
    print("Все собственные значения > 0, значит (0,0) - минимум")
else:
    print("(0,0) - не минимум")

def f(x, y):
    return x ** 2 + 3 * y ** 2 - 2 * x * y

x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

plt.figure(figsize=(8,6))
contour = plt.contour(X, Y, Z, levels=20, cmap='viridis')
plt.clabel(contour, inline=True, fontsize=8)

plt.scatter([0], [0], color='red', marker='x', s=100, linewidths=3, label='(0,0)')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Контурный график f(x, y) = x^2 + 3y^2 - 2xy')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.show()