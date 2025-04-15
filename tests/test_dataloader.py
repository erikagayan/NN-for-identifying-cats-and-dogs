import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Load data
def get_train_loader():
    dataset = datasets.ImageFolder("../dataset", transform=transform)
    return DataLoader(dataset, batch_size=32, shuffle=True)

# Test: Does the dataset load without errors
def test_dataloader_init():
    loader = get_train_loader()
    assert isinstance(loader, DataLoader)

# Test: batches of the correct size and structure
def test_dataloader_batch_shape():
    loader = get_train_loader()
    for images, labels in loader:
        assert isinstance(images, torch.Tensor)
        assert isinstance(labels, torch.Tensor)

        assert images.shape[0] <= 32
        assert images.shape[1:] == (3, 128, 128)

        assert labels.dtype == torch.int64
        break
