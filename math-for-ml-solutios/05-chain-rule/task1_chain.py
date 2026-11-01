import numpy as np

h = lambda x: x**2
g = lambda x: np.exp(x)
f = lambda x: np.sin(x)

h_d = lambda x: 2 * x
g_d = lambda x: np.exp(x)
f_d = lambda x: np.cos(x)



funcs = [h, g, f]
derivatives = [h_d, g_d, f_d]
x = 1.5

def chain(funcs, derivatives, x):
    values = [x]
    current_val = x

    for func in funcs:
        current_val = func(current_val)
        values.append(current_val)
    
    final_value = current_val

    total_derivative = 1.0

    for i in range(len(funcs) - 1, -1, -1):
        local_deriv = derivatives[i](values[i])
        total_derivative *= local_deriv
    
    return final_value, total_derivative

val, der = chain(funcs, derivatives, x)

analytic_der = np.cos(np.exp(x**2)) * 2 * x * np.exp(x**2)

print(f"Значение: {val}")
print(f"Производная (chain): {der}")
print(f"Производная (аналит.): {analytic_der}")
print(f"Разница: {abs(der - analytic_der)}")