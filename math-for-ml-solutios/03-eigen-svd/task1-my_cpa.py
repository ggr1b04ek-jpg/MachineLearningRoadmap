import numpy as np

def pca_svd(X, k):
    X_centered = X - X.mean(axis=0)
    U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
    components = Vt[:k]
    X_reduced = X_centered @ components.T
    return X_reduced, components

def pca_cov(X, k):
    X_centered = X - X.mean(axis=0)
    n = X_centered.shape[0]

    X_cov = (X_centered.T @ X_centered)/ (n - 1)
    eigvals, eigvecs = np.linalg.eigh(X_cov)

    idx = np.argsort(eigvals[::-1])
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    components = eigvecs[:, :k]
    X_reduced = X_centered @ components
    return X_reduced, components

X = np.random.randn(1000, 100)
X_2d_svd, comps_svd = pca_svd(X, 2)
X_2d_cov, comps_cov = pca_cov(X, 2)

print(f"Размеры матриц: {X_2d_svd.shape}, {X_2d_cov.shape}")
print(f"Совпадают? {np.allclose(np.abs(X_2d_cov), np.abs(X_2d_svd))}")