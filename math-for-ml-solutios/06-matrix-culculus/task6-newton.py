import numpy as np 

def f(v):
    x, y = v
    return x**2 + 3 * y**2 - 2 * x * y

def grad_f(v):
    x, y = v
    return np.array([2 * x - 2 * y, 6 * y - 2 * x])

def hess_f(v):
    return np.array([
        [2, -2],
        [-2, 6]
    ])

def newton_step(v):
    g = grad_f(v)
    H = hess_f(v)
    d = np.linalg.solve(H, g)
    return v - d

def gd(v0, lr=0.1, iters=100):
    v = v0.copy()
    history = [v.copy()]
    for _ in range(iters):
        v -= lr * grad_f(v)
        history.append(v.copy())
    return np.array(history)

v0 = np.array([5.0, -3.0])

print("Старт:", v0)
print("f(v0):", f(v0))

v_newton = newton_step(v0)
print("Ньютон, 1 шаг:", v_newton)
print("f после Ньютона:", f(v_newton))

history = gd(v0, lr=0.1, iters=100)
print("GD, 100 шагов:", history[-1])
print("f после GD:", f(history[-1]))

print("Расстояние до минимума (Ньютон):", np.linalg.norm(v_newton))
print("Расстояние до минимума (GD):    ", np.linalg.norm(history[-1]))