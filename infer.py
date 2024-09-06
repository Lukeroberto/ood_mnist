import torch
from torch import nn
from torchvision import datasets, transforms

import os
from PIL import Image
import matplotlib.pyplot as plt
import argparse

from train import Autoencoder

def png_to_mnist_tensor(image_path):
    # Open the image
    img = Image.open(image_path).convert('L')

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    tensor = transform(img)
    if tensor.mean() > 0:
        tensor = 1 - tensor

    # Ensure the tensor is of shape [1, 28, 28]
    tensor = tensor.squeeze()  # Remove any extra dimensions
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)  # Add channel dimension if needed
    return tensor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_loc")
    args = parser.parse_args()
    print(args.image_loc)

    threshold = 1000000

    weights= torch.load("ood_detector.pt")
    autoencoder = Autoencoder()
    autoencoder.load_state_dict(weights)
    autoencoder.eval()

    new_image = png_to_mnist_tensor(args.image_loc)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    mnist_test = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=256, shuffle=True)

    # Get a batch of test examples
    batch = next(iter(test_loader))
    batch_images, batch_labels = batch

    # Ensure autoencoder is in eval mode
    autoencoder.eval()

    with torch.no_grad():
        batch_embeddings = autoencoder.encode(batch_images)
        batch_embeddings = batch_embeddings.flatten(start_dim=1)
        print(batch_embeddings.shape)
        new_embedding = autoencoder.encode(new_image.unsqueeze(0))
        new_embedding = new_embedding.flatten(start_dim=1)
        print(new_embedding.shape)

        #distances = torch.cdist(new_embedding, batch_embeddings)
        distances = torch.matmul(new_embedding, batch_embeddings.T)
        closest_idx = torch.argmax(distances)

        min_distance = distances[0, closest_idx]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(new_image.squeeze(), cmap='gray')
        ax1.set_title("Given Image")
        ax1.axis('off')

        if min_distance > threshold:
            print("Out of Distribution, dist = ", min_distance)
            ax2.text(0.5, 0.5, "Out of Distribution",
                     ha='center', va='center', fontsize=16)
            ax2.axis('off')
            plt.suptitle("Out of Distribution", fontsize=16)
        else:
            closest_image = batch_images[closest_idx]
            closest_label = batch_labels[closest_idx].item()

            # Plot the closest image
            ax2.imshow(closest_image.squeeze(), cmap='gray')
            ax2.set_title(f"Closest MNIST Digit: {closest_label}")
            ax2.axis('off')

            plt.suptitle(f"Closest MNIST Digit: {closest_label}\nDistance: {min_distance.item():.4f}",
                         fontsize=16)

        # Save the plot
        plt.savefig("examples/prediction.png")
        plt.close(fig)

