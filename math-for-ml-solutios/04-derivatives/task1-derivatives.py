import numpy as np

def derivative_forward(f, x, h):
    return (f(x + h) - f(x)) / h

def derivative_central(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)

functions = [
    {
        'name': 'x^3',
        'f': lambda x: x**3,
        'df': lambda x: 3*x**2,
        'x0': 2.0
    },
    {
        'name': 'e^x',
        'f': lambda x: np.exp(x),
        'df': lambda x: np.exp(x),
        'x0': 1.0
    },
    {
        'name': 'ln(x)',
        'f': lambda x: np.log(x),
        'df': lambda x: 1 / x,
        'x0': 2.0
    },
    {
        'name': 'sin(x)',
        'f': lambda x: np.sin(x),
        'df': lambda x: np.cos(x),
        'x0': np.pi / 4
    },
    {
        'name': 'cos(x)',
        'f': lambda x: np.cos(x),
        'df': lambda x: -np.sin(x),
        'x0': np.pi / 3
    }
]

h = 1e-5

for func in functions:
    x0 = func['x0']
    analytical = func['df'](x0)
    
    forward = derivative_forward(func['f'], x0, h)
    central = derivative_central(func['f'], x0, h)
    
    error_forward = abs(forward - analytical)
    error_central = abs(central - analytical)
    
    print(f"{func['name']:<10} {analytical:<15.8f} {forward:<15.8f} {central:<15.8f} {error_forward:<15.2e} {error_central:<15.2e}")