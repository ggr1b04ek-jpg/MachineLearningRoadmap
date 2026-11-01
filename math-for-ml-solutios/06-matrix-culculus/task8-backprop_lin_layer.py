import numpy as np
import torch

np.random.seed(42)
d_in = 5
d_out = 3

x_np = np.random.randn(d_in)
W_np = np.random.randn(d_out, d_in)
b_np = np.random.randn(d_out)
grad_y_np = np.random.randn(d_out)

grad_W_numpy = grad_y_np[:, None] * x_np[None, :]
grad_b_numpy = grad_y_np
grad_x_numpy = W_np.T @ grad_y_np

print("=" * 60)
print("АНАЛИТИЧЕСКИЕ ФОРМУЛЫ (NumPy)")
print("=" * 60)
print("grad_W:\n", grad_W_numpy)
print("\ngrad_b:", grad_b_numpy)
print("\ngrad_x:", grad_x_numpy)

x_torch = torch.tensor(x_np, dtype=torch.float32, requires_grad=True)
W_torch = torch.tensor(W_np, dtype=torch.float32, requires_grad=True)
b_torch = torch.tensor(b_np, dtype=torch.float32, requires_grad=True)

y_torch = W_torch @ x_torch + b_torch

loss = (y_torch * torch.tensor(grad_y_np, dtype=torch.float32)).sum()

loss.backward()

print("\n" + "=" * 60)
print("PYTORCH (автоматическое дифференцирование)")
print("=" * 60)
print("grad_W:\n", W_torch.grad.numpy())
print("\ngrad_b:", b_torch.grad.numpy())
print("\ngrad_x:", x_torch.grad.numpy())

print("\n" + "=" * 60)
print("СРАВНЕНИЕ")
print("=" * 60)
print("grad_W совпадают?", np.allclose(grad_W_numpy, W_torch.grad.numpy()))
print("grad_b совпадают?", np.allclose(grad_b_numpy, b_torch.grad.numpy()))
print("grad_x совпадают?", np.allclose(grad_x_numpy, x_torch.grad.numpy()))