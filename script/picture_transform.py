"""
This script does the conversion, loading images from folders and creates a loader
"""

from script.dataloader import get_train_loader
from utils.visual import show_images

train_loader, classes = get_train_loader()

images, labels = next(iter(train_loader))
show_images(images, labels, classes)
