import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from fontTools.cffLib import width

from augmentations_basics.datasets import CustomImageDataset

dataset = CustomImageDataset("data/train")
class_counts = {}
sizes = []

for c in dataset.classes:
    class_dir = f"data/train/{c}"
    files = [f for f in os.listdir(class_dir) if f.lower().endswith((".png", ".jpg"))]
    class_counts[c] = len(files)

    for f in files:
        path = os.path.join(class_dir, f)
        with Image.open(path) as img:
            sizes.append(img.size)

print("Количество изображений в каждом классе:")
for c, n in class_counts.items():
    print(f"{c}: {n}")

print("\nРазмеры изображений:")
print(f"Минимальный размер: {min(sizes, key=lambda x: x[0] * x[1])}")
print(f"Максимальный размер: {max(sizes, key=lambda x: x[0] * x[1])}")
width, height = zip(*sizes)
print(f"Средний размер: {np.mean(width):.2f} x {np.mean(height):.2f}")

plt.figure(figsize=(8,6))
y_pos = np.arange(len(class_counts.keys()))
plt.barh(y_pos, class_counts.values())
plt.yticks(y_pos, class_counts.keys())
plt.title("Количество изображений по классам")
plt.xlabel("Count")

plt.tight_layout()
plt.savefig("results/task3_counts.png")
plt.close()

plt.figure(figsize=(8,6))
plt.hist(width, bins=20, alpha=0.5, label="width")
plt.hist(height, bins=20, alpha=0.5, label="height")
plt.title("Высота и ширина изображений")
plt.xlabel("Pixels")
plt.ylabel("Count")
plt.legend()

plt.tight_layout()
plt.savefig("results/task3_sizes.png")
plt.close()