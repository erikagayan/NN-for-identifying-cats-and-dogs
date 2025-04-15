"""
This script does the conversion, loading images from folders and creates a loader
"""

import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Loading images from folders and Creating bootloaders
train_data = datasets.ImageFolder('../dataset', transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)


# 📌 5. Проверка одного батча
for images, labels in train_loader:
    print("Размер батча:", images.shape)  # [32, 3, 128, 128]
    print("Метки классов:", labels)       # тензор с номерами классов
    break

# 📌 6. Отображение первых 6 изображений
def show_images(images, labels, classes):
    fig, axes = plt.subplots(1, 6, figsize=(12, 4))
    for i in range(6):
        img = images[i] * 0.5 + 0.5  # отменяем нормализацию
        axes[i].imshow(img.permute(1, 2, 0))
        axes[i].set_title(classes[labels[i]])
        axes[i].axis("off")
    plt.show()

# Получаем названия классов и отображаем картинки
classes = train_data.classes  # ['cat', 'dog']
show_images(images, labels, classes)
