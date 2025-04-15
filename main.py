"""
This is the main project file where:
- the dataset is loaded
- the model is created
- the loss function and optimizer are selected
"""

from models.cnn import get_simple_cnn
from script.dataloader import get_train_loader
import torch
from torch import nn, optim

train_loader, classes = get_train_loader()


model = get_simple_cnn()

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 5


for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    for batch_idx, (images, labels) in enumerate(train_loader):
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"[{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}")

print("\n✅ Обучение завершено!")
