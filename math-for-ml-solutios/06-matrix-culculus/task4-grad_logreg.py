import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def grad_logreg(X, y, w):
    return X.T @ (sigmoid(X @ w) - y)

def predict(X, w):
    probs = sigmoid(X @ w)
    return (probs >= 0.5).astype(int)

def gd(X, y, grad_func, lr=0.1, iters=1000):
    w = np.zeros(X.shape[1])
    for it in range(iters):
        w -= lr * grad_func(X, y, w)
    return w

iris = load_iris()
X_full = iris.data 
y_full = iris.target

mask = (y_full == 0) | (y_full == 1)
X = X_full[mask]
y = y_full[mask]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = np.hstack([np.ones((X_train_scaled.shape[0], 1)), X_train_scaled])
X_test_scaled = np.hstack([np.ones((X_test_scaled.shape[0], 1)), X_test_scaled])

w = gd(X_train_scaled, y_train, grad_logreg, lr=0.1, iters=1000)

y_pred = predict(X_test_scaled, w)
accuracy = (y_pred == y_test).mean()

print(f"Accuracy на test: {accuracy:.4f}")
print(f"{'Предсказания':<17} {y_pred}")
print(f"{'Истинные метки':<17} {y_test}")