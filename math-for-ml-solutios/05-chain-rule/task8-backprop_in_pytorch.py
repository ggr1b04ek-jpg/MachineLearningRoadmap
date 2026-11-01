import torch

x = torch.tensor(2.0)
y = torch.tensor(1.0)
w = torch.tensor(0.5, requires_grad=True)
b = torch.tensor(0.1, requires_grad=True)
lr = 0.1
steps = 1000

print(f"Начальное предсказание: {torch.sigmoid(w*x + b).item():.4f}")

for i in range(steps):
    z = w * x + b
    a = torch.sigmoid(z)
    L = (a - y)**2

    L.backward()

    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

        w.grad.zero_()
        b.grad.zero_()
    
    if (i + 1) % 200 == 0:
        print(f"Шаг {i+1}: Loss = {L.item():.6f}, Предсказание = {a.item():.4f}")

print(f"\nИтоговое предсказание: {torch.sigmoid(w*x + b).item():.6f}")