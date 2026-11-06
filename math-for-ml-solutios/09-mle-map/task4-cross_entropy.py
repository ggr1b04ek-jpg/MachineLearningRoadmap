import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data[:100, :2]
y = iris.target[:100]

X = (X - X.mean(axis=0)) / X.std(axis=0)

X_b = np.c_[X, np.ones(X.shape[0])]

n_samples, n_features = X_b.shape

def sigmoid(z):
  z = np.clip(z, -250, 250)
  return 1 / (1 + np.exp(-z))

def compute_loss(X, y, w):
  y_pred = sigmoid(X @ w)

  eps = 1e-15
  y_pred = np.clip(y_pred, eps, 1 - eps)
  n = len(y_pred)

  loss = -(1 / n) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
  return loss

def compute_gradients(X, y, w):
  y_pred = sigmoid(X @ w)
  error = y_pred - y
  N = len(y)

  grad = (1 / N) * X.T @ error
  return grad

w = np.zeros(n_features)
lr = 0.5
epochs = 200

loss_history = []

for epoch in range(epochs):
  grad = compute_gradients(X_b, y, w)

  w -= grad * lr

  loss = compute_loss(X_b, y, w)
  loss_history.append(loss)

print(f"Итоговые веса (w1, w2, b): {w}")
print(f"Итоговый Loss: {loss_history[-1]:.4f}")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(loss_history, color='blue')
plt.title('Падение функции потерь (Cross-Entropy)')
plt.xlabel('Эпоха')
plt.ylabel('Loss')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(X[y==0, 0], X[y==0, 1], color='red', label='Класс 0', alpha=0.7)
plt.scatter(X[y==1, 0], X[y==1, 1], color='blue', label='Класс 1', alpha=0.7)

x1_vals = np.array([X[:, 0].min() - 1, X[:, 0].max() + 1])
x2_vals = -(w[0] * x1_vals + w[2]) / w[1]
plt.plot(x1_vals, x2_vals, 'g-', linewidth=2, label='Decision Boundary')

plt.xlabel('Признак 1 (норм.)')
plt.ylabel('Признак 2 (норм.)')
plt.title('Разделение двух классов Iris')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()