import numpy as np

def softmax(z):
    z_shifted = z - np.max(z)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z)

def jacobian_num(z, h=1e-5):
    n = len(z)
    m = len(softmax(z))
    jac = np.zeros((n, m))

    for i in range(n):
            z_plus = z.copy(); z_plus[i] += h
            z_minus = z.copy(); z_minus[i] -= h

            s_plus = softmax(z_plus)
            s_minus = softmax(z_minus)

            jac[:, i] = (s_plus - s_minus) / (2 * h)
    
    return jac
    

def jacobian_ana(z):
    s = softmax(z)
    return np.diag(s) - np.outer(s, s)

z = np.random.randn(3)

print(np.allclose(jacobian_ana(z), jacobian_num(z)))