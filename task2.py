from torchvision import transforms
from augmentations_basics.datasets import CustomImageDataset
from augmentations_basics.utils import show_multiple_augmentations
from augmentations_basics.extra_augs import AddGaussianNoise, Posterize
from torchvision.transforms import functional as F
import random


class RandomBlur:
    """Случайное размытие"""

    def __init__(self, radius_range=(3, 5), sigma_range=(0.1, 2.0)):
        self.radius_range = radius_range
        self.sigma_range = sigma_range

    def __call__(self, img):
        kernel_size = random.choice([k for k in range(self.radius_range[0], self.radius_range[1] + 1) if k % 2 == 1])
        sigma = random.uniform(self.sigma_range[0], self.sigma_range[1])

        return F.gaussian_blur(img, [kernel_size, kernel_size], [sigma, sigma])


class RandomPerspective:
    """Случайная перспектива"""

    def __init__(self, distortion=0.5):
        self.distortion = distortion
    
    def __call__(self, img):
        _, height, width = img.shape
        start = [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ]

        m = int(min(height, width) * self.distortion)
        end = [[x + random.randint(-m, m), y + random.randint(-m, m)] for x, y in start]
        
        return F.perspective(img, start, end)


class RandomBrightnessContrast:
    """Случайная яркость и контраст"""

    def __init__(self, brightness_range=(0.5, 1.5), contrast_range=(0.5, 1.5)):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range

    def __call__(self, img):
        brightness = random.uniform(self.brightness_range[0], self.brightness_range[1])
        contrast = random.uniform(self.contrast_range[0], self.contrast_range[1])

        img = F.adjust_brightness(img, brightness)
        img = F.adjust_contrast(img, contrast)

        return img


dataset = CustomImageDataset("data/train", transform=transforms.ToTensor(), target_size=(224, 224))

augmentations = transforms.Compose([
    RandomBlur(radius_range=(3, 5), sigma_range=(0.1, 2.0)),
    RandomPerspective(distortion=0.5),
    RandomBrightnessContrast(brightness_range=(0.5, 1.5), contrast_range=(0.5, 1.5)),
    AddGaussianNoise(mean=0., std=0.1),
    Posterize(bits=2),
])

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

    for augmentation in augmentations.transforms:
        augmented_img = augmentation(image)
        augmented_imgs.append(augmented_img)
        titles.append(augmentation.__class__.__name__)

    show_multiple_augmentations(image, augmented_imgs, titles)