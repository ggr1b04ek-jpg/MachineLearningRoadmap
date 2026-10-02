import numpy as np

def my_pinv(A):

    U, S, VT = np.linalg.svd(A, full_matrices=False)

    threshold = 1e-10
    S_plus = np.array([1/s if s > threshold else 0 for s in S])

    Sigma_plus = np.diag(S_plus)

    A_pinv = VT.T @ Sigma_plus @ U.T

    return A_pinv

A = np.random.randn(4, 3)

A_pinv_manual = my_pinv(A)

A_pinv_np = np.linalg.pinv(A)

print(f"Совпадают? {np.allclose(A_pinv_manual, A_pinv_np)}")

result = A @ A_pinv_manual @ A
print("A @ A_pinv @ A = A ?", np.allclose(result, A))