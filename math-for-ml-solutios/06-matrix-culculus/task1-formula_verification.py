import numpy as np

tests = [
    {
        "name": "1",
        "setup" : lambda: (np.random.randn(3), np.random.randn(3)),
        "func" : lambda x, a: a.T @ x,
        "grad" : lambda x, a: a
    },
    {
        "name": "2",
        "setup" : lambda: (np.random.randn(3), np.random.randn(3, 3)),
        "func" : lambda x, a: x.T @ a @ x,
        "grad" : lambda x, a: (a + a.T) @ x
    },
    {
        "name": "3",
        "setup" : lambda: (np.random.randn(3), (lambda M: (M + M.T)/2)(np.random.randn(3, 3))),
        "func" : lambda x, a: x.T @ a @ x,
        "grad" : lambda x, a: 2 * a @ x
    },
    {
        "name": "4",
        "setup" : lambda: (np.random.randn(3), np.random.randn(3, 3)),
        "func" : lambda x, a: a @ x,
        "grad" : lambda x, a: a
    },
    {
        "name": "5",
        "setup" : lambda: (np.random.randn(3), np.eye(3)),
        "func" : lambda x, a: x.T @ a @ x,
        "grad" : lambda x, a: 2 * x
    },
    {
        "name": "6",
        "setup" : lambda: (np.random.randn(3, 3), np.random.randn(3), np.random.randn(3)),
        "func" : lambda w, a, b: a.T @ w @ b,
        "grad" :  lambda w, a, b: np.outer(a, b)
    },
    {
        "name": "7",
        "setup" : lambda: (np.random.randn(3, 3), np.random.randn(3, 3)),
        "func" : lambda w, a: np.trace(w.T @ a),
        "grad" : lambda w, a: a
    },
    {
        "name": "8",
        "setup" : lambda: (np.random.randn(3, 3), np.random.randn(3, 3)),
        "func" : lambda w, a: np.trace(w @ a @ w.T),
        "grad" : lambda w, a: w @ (a + a.T)
    },
    {
        "name": "9",
        "setup" : lambda: (np.random.randn(3, 3), np.random.randn(3, 3), np.random.randn(3, 3)),
        "func" : lambda w, x, y: np.sum((x @ w - y)**2),
        "grad" : lambda w, x, y: 2 * x.T @ (x @ w - y)
    },
    {
        "name": "10",
        "setup" : lambda: (np.random.randn(3, 3), np.random.randn(3, 3)),
        "func" : lambda w, a: np.trace(a @ w),
        "grad" : lambda w, a: a.T
    }
]

def numerical_grad(f, x, h=1e-5):
    original_shape = x.shape
    x_flat = x.flatten()
    grad_flat = np.zeros_like(x_flat)

    for i in range(len(x_flat)):
        x_plus = x_flat.copy(); x_plus[i] += h
        x_minus = x_flat.copy(); x_minus[i] -= h

        val_plus = f(x_plus.reshape(original_shape))
        val_minus = f(x_minus.reshape(original_shape))

        grad_flat[i] = (val_plus - val_minus) / (2 * h)
    
    return grad_flat.reshape(original_shape)

passed = 0



for t in tests:
    data = t["setup"]()
    f = lambda v, consts=data[1:]: t["func"](v, *consts)
    h = 1e-5

    true_grad = t["grad"](*data)
    test_val = f(data[0])
    
    if np.isscalar(test_val) or (hasattr(test_val, 'ndim') and test_val.ndim == 0):
        num_grad = numerical_grad(f, data[0])
        
        max_diff = np.max(np.abs(num_grad - true_grad))
        is_ok = max_diff < 1e-5
    else:
        m = len(test_val)
        n = len(data[0].flatten())
        jacobian_num = np.zeros((m, n))
        x_flat = data[0].flatten()
        orig_shape = data[0].shape
        
        for i in range(n):
            x_plus = x_flat.copy(); x_plus[i] += h
            x_minus = x_flat.copy(); x_minus[i] -= h
            
            val_p = f(x_plus.reshape(orig_shape))
            val_m = f(x_minus.reshape(orig_shape))
            
            jacobian_num[:, i] = (val_p - val_m) / (2 * h)
            
        num_grad = jacobian_num.reshape(m, *orig_shape[1:]) if len(orig_shape) > 1 else jacobian_num
        
        max_diff = np.max(np.abs(num_grad - true_grad))
        is_ok = max_diff < 1e-5

    status = "passed" if is_ok else "failed"
    if is_ok: passed += 1

    print(f"{t['name']:3s} | diff: {max_diff:.2e} | {status}")

print(f"\nИтог: {passed}/{len(tests)} формул верны.")