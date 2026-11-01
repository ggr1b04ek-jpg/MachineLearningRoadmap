import numpy as np
import random

data = np.load('C:/Users/user/MachineLearningRoadmap/math-for-ml-solutios/05-chain-rule/mnist.npz')
train_images = data['x_train']
train_labels = data['y_train']
test_images = data['x_test']
test_labels = data['y_test']

train_images = train_images.reshape((60000, 784))
test_images = test_images.reshape((10000, 784))

train_images = train_images / 255.0
test_images = test_images / 255.0

w1 = np.random.randn(784, 64) * np.sqrt(2 / 784)
b1 = np.zeros(64)
w2 = np.random.randn(64, 10) * np.sqrt(2 / 64)
b2 = np.zeros(10)

h = 1e-5
index_i = random.randint(0, 783)
index_j = random.randint(0, 63)

print(f"Проверка i, j:{index_i}, {index_j}")

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def forward(x):
    z1 = x @ w1 + b1
    a1 = np.maximum(0, z1)
    z2 = a1 @ w2 + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2

def compute_loss(a2, y):
    correct_probs = a2[np.arange(len(y)), y]
    loss = -np.mean(np.log(correct_probs))
    return loss

def backward(x, y, z1, a1, z2, a2):
    y_one_hot = np.eye(10)[y]
    n = y_one_hot.shape[0]

    dz2 = a2 - y_one_hot

    dw2 = a1.T @ dz2 / n
    db2 = np.sum(dz2, axis=0) / n

    da1 = dz2 @ w2.T

    dz1 = da1 * (z1 > 0)

    dw1 = x.T @ dz1 / n
    db1 = np.sum(dz1, axis=0) / n

    return dw1, db1, dw2, db2

original_val = w1[index_i, index_j]

w1[index_i, index_j] = original_val + h
_, _, _, a2_plus = forward(train_images[:100])
loss_plus = compute_loss(a2_plus, train_labels[:100])

w1[index_i, index_j] = original_val - h
_, _, _, a2_minus = forward(train_images[:100])
loss_minus = compute_loss(a2_minus, train_labels[:100])

w1[index_i, index_j] = original_val

grad_num = (loss_plus - loss_minus) / (2 * h)

z1, a1, z2, a2 = forward(train_images[:100])
dw1, _, _, _ = backward(train_images[:100], train_labels[:100], z1, a1, z2, a2)
grad_ana = dw1[index_i, index_j]

print(f"Численный градиент: {grad_num}")
print(f"Аналитический градиент: {grad_ana}")
print(f"Разница: {abs(grad_num - grad_ana)}")