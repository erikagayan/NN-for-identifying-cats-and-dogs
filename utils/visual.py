import matplotlib.pyplot as plt


def show_images(images, labels, classes, unnormalize=True):
    """Display 6 images from a batch with class labels"""
    fig, axes = plt.subplots(1, 6, figsize=(12, 4))
    for i in range(6):
        img = images[i]
        if unnormalize:
            img = img * 0.5 + 0.5  # undo normalization
        img = img.permute(1, 2, 0)  # CHW -> HWC
        axes[i].imshow(img)
        axes[i].set_title(classes[labels[i]])
        axes[i].axis("off")
    plt.show()
