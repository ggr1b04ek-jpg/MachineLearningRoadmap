import numpy as np

data = np.load('C:/Users/user/MachineLearningRoadmap/math-for-ml-solutios/05-chain-rule/mnist.npz')
train_images = data['x_train'].reshape((60000, 784)) / 255.0
train_labels = data['y_train']
test_images = data['x_test'].reshape((10000, 784)) / 255.0
test_labels = data['y_test']

layer_sizes = [784, 128, 128, 128, 128, 10]
num_layers = len(layer_sizes) - 1

weights = []
biases = []
for i in range(num_layers):
    n_in = layer_sizes[i]
    n_out = layer_sizes[i+1]
    w = np.random.randn(n_in, n_out) * np.sqrt(2 / n_in)
    b = np.zeros(n_out)
    weights.append(w)
    biases.append(b)

activation_type = 'sigmoid' 

def activation(z):
    if activation_type == 'relu':
        return np.maximum(0, z)
    else:
        return 1 / (1 + np.exp(-z))

def activation_deriv(z, a):
    if activation_type == 'relu':
        return (z > 0).astype(float)
    else:
        return a * (1 - a)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def forward(x):
    activations = [x]
    pre_activations = []
    
    current_input = x
    for i in range(num_layers):
        z = current_input @ weights[i] + biases[i]
        pre_activations.append(z)
        
        if i == num_layers - 1:
            a = softmax(z)
        else:
            a = activation(z)
            
        activations.append(a)
        current_input = a
        
    return activations, pre_activations

def compute_loss(a_final, y):
    correct_probs = a_final[np.arange(len(y)), y]
    return -np.mean(np.log(correct_probs + 1e-9))

def backward(x, y, activations, pre_activations):
    n = x.shape[0]
    grads_w = [None] * num_layers
    grads_b = [None] * num_layers
    
    dz = activations[-1] - np.eye(10)[y]
    
    for i in range(num_layers - 1, -1, -1):
        grads_w[i] = activations[i].T @ dz / n
        grads_b[i] = np.sum(dz, axis=0) / n
        
        if i > 0:
            da = dz @ weights[i].T
            dz = da * activation_deriv(pre_activations[i-1], activations[i])
            
    return grads_w, grads_b

def train_model(epochs=50, lr=0.01):
    for epoch in range(epochs):
        acts, pre_acts = forward(train_images)
        loss = compute_loss(acts[-1], train_labels)
        
        dw, db = backward(train_images, train_labels, acts, pre_acts)
        
        for i in range(num_layers):
            weights[i] -= lr * dw[i]
            biases[i] -= lr * db[i]
            
        if (epoch + 1) % 10 == 0:
            print(f"Эпоха {epoch+1}, Loss: {loss:.4f}")

print(f"Используемая активация: {activation_type}")
train_model(epochs=100, lr=0.01)

_, test_acts = forward(test_images)
preds = np.argmax(test_acts[-1], axis=1)
acc = np.mean(preds == test_labels)
print(f"Точность: {acc * 100:.2f}%")