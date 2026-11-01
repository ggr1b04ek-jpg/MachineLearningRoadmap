import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from scipy.sparse import random as sparse_random
import time

X_sparse = sparse_random(1000, 5000, density=0.01, format='csr', random_state=42)

N = 10

start = time.time()
svd = TruncatedSVD(n_components=N, random_state=42)
X_svd = svd.fit_transform(X_sparse)
time_svd = time.time() - start
print(f"TruncatedSVD время: {time_svd:.4f} сек")

start = time.time()
pca = PCA(n_components=N, random_state=42)
X_pca = pca.fit_transform(X_sparse.toarray()) 
time_pca = time.time() - start
print(f"PCA время:       {time_pca:.4f} сек")

X_centered = X_sparse.toarray() - X_sparse.toarray().mean(axis=0)
svd_manual = TruncatedSVD(n_components=N, random_state=42)
X_svd_centered = svd_manual.fit_transform(X_centered)

correlation = np.abs(np.corrcoef(X_pca[:, 0], X_svd_centered[:, 0])[0, 1])
print(f"\nКорреляция 1-й компоненты (PCA vs Centred SVD): {correlation:.5f}")