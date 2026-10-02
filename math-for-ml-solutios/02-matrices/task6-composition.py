import numpy as np

theta = np.radians(30)
R = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
S = np.array([[2, 0],[0, 1]])

M_composed = S @ R

point = np.array([1, 1])

point_after_R = R @ point
point_step_by_step = S @ point_after_R

point_composed = M_composed @ point

print(f"Точка после поворота: {point_after_R}")
print(f"Точка после масштабирования: {point_step_by_step}")
print(f"Точка после применения итоговой матрицы: {point_composed}")
print(f"Преобразования равны? {np.allclose(point_step_by_step, point_composed)}")