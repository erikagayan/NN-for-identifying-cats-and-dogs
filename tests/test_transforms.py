import torch
from PIL import Image
from torchvision import transforms


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


# Tensor shape test
def test_output_shape():
    img = Image.new("RGB", (256, 256))
    tensor = transform(img)
    assert tensor.shape == (3, 128, 128)

# Range test (after normalization)
def test_range_after_normalization():
    img = Image.new("RGB", (128, 128), (127, 127, 127))
    tensor = transform(img)
    assert torch.all(tensor >= -1) and torch.all(tensor <= 1)
