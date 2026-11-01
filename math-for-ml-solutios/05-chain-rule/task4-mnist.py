import numpy as np

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

def train_step(x, y):
    z1, a1, z2, a2 = forward(x)

    loss = compute_loss(a2, y)

    dw1, db1, dw2, db2 = backward(x, y, z1, a1, z2, a2)

    global w1, b1, w2, b2, lr
    w1 -= lr * dw1
    b1 -= lr * db1
    w2 -= lr * dw2
    b2 -= lr * db2

    return loss

def train_model(epochs=50, lr=0.1, batch_size=128):
    n_samples = train_images.shape[0]

    for epoch in range(epochs):
        indices = np.random.permutation(n_samples)
        shuffled_images = train_images[indices]
        shuffled_labels = train_labels[indices]

        for i in range(0, n_samples, batch_size):
            x_batch = shuffled_images[i:i+batch_size]
            y_batch = shuffled_labels[i:i+batch_size]
            train_step(x_batch, y_batch)
        
        _, _, _, a2 = forward(train_images)
        loss = compute_loss(a2, train_labels)

        if (epoch + 1) % 20 == 0:
            print(f"Эпоха {epoch+1}/{epochs}, loss: {loss:.4f}")
        
lr = 0.1
train_model(epochs=200, lr=lr, batch_size=128)

_, _, _, test_probs = forward(test_images)

predicted_digits = np.argmax(test_probs, axis=1)

accuracy = np.mean(predicted_digits == test_labels)

print(f"Точность на тестовой выборе: {accuracy * 100:.2f}%")