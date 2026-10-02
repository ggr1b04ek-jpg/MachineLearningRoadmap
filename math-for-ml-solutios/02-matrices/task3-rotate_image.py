import numpy as np
import matplotlib.pyplot as plt

theta = np.pi / 6
cos_a = np.cos(theta)
sin_a = np.sin(theta)
img = plt.imread(r"C:\Users\user\MachineLearningRoadmap\math-for-ml-solutios\02-matrices\task3-img.jpg")
img_array = np.array(img)
height, width = img_array.shape[:2]

x = np.arange(width) - width / 2
y = np.arange(height) - height / 2
yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')


x_new = (xx - width / 2) * cos_a - (yy - height / 2) * sin_a
y_new = (xx - width / 2) * sin_a + (yy - height / 2) * cos_a

x_final = x_new + width / 2
y_final = y_new + height / 2

x_indices = np.round(x_final).astype(int)
y_indices = np.round(y_final).astype(int)

new_img = np.zeros_like(img)

valid_mask = (x_indices >= 0) & (x_indices < width) & \
    (y_indices >= 0) & (y_indices < height)

new_img[y_indices[valid_mask], x_indices[valid_mask]] = img[yy[valid_mask], xx[valid_mask]]

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Оригинал")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(new_img)
plt.title("Поворот на 30 градусов")
plt.axis('off')

plt.show()