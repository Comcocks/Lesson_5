import time
from augmentations_basics.datasets import CustomImageDataset
import tracemalloc
from torchvision import transforms
import matplotlib.pyplot as plt


sizes = [64, 128, 224, 512]
datasets = [
    CustomImageDataset("data/train", transform=None, target_size=(sizes[0], sizes[0])),
    CustomImageDataset("data/train", transform=None, target_size=(sizes[1], sizes[1])),
    CustomImageDataset("data/train", transform=None, target_size=(sizes[2], sizes[2])),
    CustomImageDataset("data/train", transform=None, target_size=(sizes[3], sizes[3]))
]

# Тест размеров с помощью размытия
times, mems = [], []
for dataset in datasets:
    tracemalloc.start()
    start = time.time()

    for i in range(100):
        img, _ = dataset[i]
        _ = transforms.GaussianBlur((5, 5))(img)

    end = time.time()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    times.append(end - start)
    mems.append(peak / 1024 / 1024)

# Сохранение графиков
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(sizes, times, marker="s")
plt.xticks(sizes)
plt.title("Время выполнения")
plt.xlabel("Image size (px)")
plt.ylabel("Time (sec)")

plt.subplot(1, 2, 2)
plt.plot(sizes, mems, marker="s")
plt.xticks(sizes)
plt.title("Использование памяти")
plt.xlabel("Image size (px)")
plt.ylabel("Memory (Mb)")

plt.tight_layout()
plt.savefig("results/time_memory_dependence.png")
plt.close()