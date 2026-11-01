import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from PIL import Image

DATA_DIR = r'C:\Users\user\Downloads\archive\lfw_funneled' 

if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(
        f"Папка не найдена: {DATA_DIR}\n"
        "Проверьте, что вы распаковали lfw-funneled.tgz из архива."
    )

MIN_FACES = 70
IMG_SIZE = (62, 47) 

images = []
labels = []
target_names = sorted(os.listdir(DATA_DIR))

for label_idx, person_name in enumerate(target_names):
    person_dir = os.path.join(DATA_DIR, person_name)
    
    if not os.path.isdir(person_dir):
        continue
    
    img_files = [f for f in os.listdir(person_dir) if f.endswith('.jpg')]
    
    if len(img_files) < MIN_FACES:
        continue
        
    for img_file in img_files:
        img_path = os.path.join(person_dir, img_file)
        img = Image.open(img_path).convert('L').resize(IMG_SIZE)
        images.append(np.array(img, dtype=np.float64).flatten())
        labels.append(label_idx)

X = np.array(images)
y = np.array(labels)
h, w = IMG_SIZE[1], IMG_SIZE[0]


pca = PCA(n_components=50, whiten=True, random_state=42)
pca.fit(X)

fig, axes = plt.subplots(4, 4, figsize=(18,18))
axes = axes.flatten()

for i in range(16):
    eigenface = pca.components_[i].reshape(h, w)

    axes[i].imshow(eigenface, cmap='gray')
    axes[i].set_title(f'Компонента {i+1}', fontsize=12)
    axes[i].axis('off')

plt.suptitle('Первые 16 Eigenfaces (главные компоненты PCA)',
            fontsize=18, y=1.02)
plt.tight_layout()
plt.show()

mean_face = pca.mean_.reshape(h, w)
plt.figure(figsize=(6,6))
plt.imshow(mean_face, cmap='gray')
plt.title('Среднее лицо датасета', fontsize=12)
plt.axis('off')
plt.tight_layout()
plt.show()