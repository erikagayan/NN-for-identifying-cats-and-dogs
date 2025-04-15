"""
This script does the conversion, loading images from folders and creates a loader
"""

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils.visual import show_images

# Transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Loading images from folders and Creating bootloaders
train_data = datasets.ImageFolder('dataset', transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# Get the batch and classes
images, labels = next(iter(train_loader))
classes = train_data.classes

# Showing images
show_images(images, labels, classes)
