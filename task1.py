from torchvision import transforms
from augmentations_basics.datasets import CustomImageDataset
from augmentations_basics.utils import show_multiple_augmentations


dataset = CustomImageDataset("data/train", transform=transforms.ToTensor(), target_size=(224, 224))

# Инициализация аугментаций
augs_dict = transforms.Compose({
    "RandomHorizontalFlip": transforms.RandomHorizontalFlip(p=1),
    "RandomCrop": transforms.RandomCrop(200, padding=10),
    "ColorJitter": transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
    "RandomRotation": transforms.RandomRotation(45),
    "RandomGrayscale": transforms.RandomGrayscale(p=1),
})

# Все аугментации вместе
augmentations = transforms.Compose(augs_dict.transforms.values())

# Создание списка изображений
images = {}
for img, label in dataset:
    if label not in images.keys():
        images[label] = img
    if len(images) == 5:
        break

for label, image in images.items():
    classes = dataset.get_class_names()
    print(f"Класс: {classes[label]}")

    augmented_imgs = []
    titles = []

    # Применение аугментаций к изображению
    for (name, augmentation) in augs_dict.transforms.items():
        augmented_img = augmentation(image)
        augmented_imgs.append(augmented_img)
        titles.append(name)

    # Все аугментации вместе
    augmented_imgs.append(augmentations(image))
    titles.append("AllAugmentations")

    # Визуализация аугментаций для одного изображения из класса
    show_multiple_augmentations(image, augmented_imgs, titles)