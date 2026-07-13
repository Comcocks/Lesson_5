import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from augmentations_basics.datasets import CustomImageDataset

dataset = CustomImageDataset("data/train")
class_counts = {}
widths = []
heights = []

for c in dataset.classes:
    class_dir = f"data/train/{c}"
    files = [f for f in os.listdir(class_dir) if f.lower().endswith((".png", ".jpg"))]
    class_counts[c] = len(files)

    for f in files:
        path = os.path.join(class_dir, f)
        with Image.open(path) as img:
            widths.append(img.width)
            heights.append(img.height)

print("Количество изображений в каждом классе:")
for c, n in class_counts.items():
    print(f"{c}: {n}")

widths_np = np.array(widths)
heights_np = np.array(heights)
print("\nРазмеры изображений:")
print(f"Максимальная ширина: {widths_np.max()}, Максимальная высота: {heights_np.max()}")
print(f"Минимальная ширина: {widths_np.min()}, Минимальная высота: {heights_np.min()}")
print(f"Средняя ширина: {widths_np.mean():.2f}, Средняя высота: {heights_np.mean():.2f}")

plt.figure(figsize=(8,6))
y_pos = np.arange(len(class_counts.keys()))
plt.barh(y_pos, class_counts.values())
plt.yticks(y_pos, class_counts.keys())
plt.title("Количество изображений по классам")
plt.xlabel("Count")

plt.tight_layout()
plt.savefig("results/images_counts.png")
plt.close()

plt.figure(figsize=(8,6))
plt.hist(widths_np, bins=20, alpha=0.5, label="width")
plt.hist(heights_np, bins=20, alpha=0.5, label="height")
plt.title("Высота и ширина изображений")
plt.xlabel("Pixels")
plt.ylabel("Count")
plt.legend()

plt.tight_layout()
plt.savefig("results/images_sizes.png")
plt.close()