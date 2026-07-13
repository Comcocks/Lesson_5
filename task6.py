import torch
from matplotlib import pyplot as plt
from torchvision import transforms, models
from torch.utils.data import DataLoader
from augmentations_basics.datasets import CustomImageDataset


def run_epoch(model, data_loader, loss_fn, optimizer=None, is_val=False):
    if is_val:
        model.eval()

    model.train()
    total_loss = 0
    acc = 0
    total = 0
    for images, labels in data_loader:
        if not is_val:
            optimizer.zero_grad()

        out = model(images)
        loss = loss_fn(out, labels)

        if not is_val:
            loss.backward()
            optimizer.step()

        pred = out.argmax(dim=1, keepdim=True)
        total_loss += loss.item()
        acc += pred.eq(labels.view_as(pred)).sum().item()
        total += labels.size(0)

    return total_loss / len(data_loader), acc / total


# Подготовка датасета
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
train_dataset = CustomImageDataset('data/train', transform=transform)
val_dataset = CustomImageDataset('data/val', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Загрузка предобученной модели
model = models.resnet18(weights='IMAGENET1K_V1')
model.fc = torch.nn.Linear(model.fc.in_features, len(train_dataset.get_class_names()))

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()

# Обучение
train_losses, train_accuracies = [], []
val_losses, val_accuracies = [], []
for epoch in range(10):
    train_loss, train_accuracy = run_epoch(model, train_loader, loss_fn, optimizer)
    val_loss, val_accuracy = run_epoch(model, val_loader, loss_fn, is_val=True)

    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

# Отображение графиков
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train', marker='o')
plt.plot(val_losses, label='Test', marker='s')
plt.xticks(range(10))
plt.title('Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train', marker='o')
plt.plot(val_accuracies, label='Test', marker='s')
plt.xticks(range(10))
plt.title('Accuracies')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig("results/task6.png")
plt.close()