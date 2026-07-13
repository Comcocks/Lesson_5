import os
from augmentations_basics.datasets import CustomImageDataset
from torchvision import transforms


class AugmentationPipeline:
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def add_augmentation(self, name, augmentation):
        self.augmentations[name] = augmentation

    def remove_augmentation(self, name):
        if name in self.augmentations:
            del self.augmentations[name]

    def apply(self, image):
        for name, augmentation in self.augmentations.items():
            image = augmentation(image)
        return image

    def get_augmentations(self):
        return self.augmentations


dataset = CustomImageDataset("data/train", transform=transforms.ToTensor(), target_size=(224, 224))

light_configuration = AugmentationPipeline({
    "RandomHorizontalFlip": transforms.RandomHorizontalFlip()
})

medium_configuration = AugmentationPipeline({
    "RandomHorizontalFlip": transforms.RandomHorizontalFlip(),
    "RandomRotation": transforms.RandomRotation(degrees=15),
    "RandomResizedCrop": transforms.RandomResizedCrop(224, scale=(0.8, 1.0))
})

heavy_configuration = AugmentationPipeline({
    "RandomHorizontalFlip": transforms.RandomHorizontalFlip(),
    "RandomRotation": transforms.RandomRotation(degrees=15),
    "RandomResizedCrop": transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    "RandomPerspective": transforms.RandomPerspective(p=1),
    "GaussianBlur": transforms.GaussianBlur(5)
})

configurations = {
    "Light": light_configuration,
    "Medium": medium_configuration,
    "Heavy": heavy_configuration
}

results = []
classes = dataset.get_class_names()
idx = 0
for name, config in configurations.items():
    for image, label in dataset:
        augmented_image = config.apply(image)
        os.makedirs(f"results/task4/{classes[label]}", exist_ok=True)

        pil = transforms.ToPILImage()(augmented_image)
        pil.save(f"results/task4/{classes[label]}/{idx}_{name}.jpg")
        idx = idx + 1