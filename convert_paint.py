import torch
import os
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def png_to_mnist_tensor(image_path):
    # Open the image
    img = Image.open(image_path).convert('L')

    # Define the transformation pipeline
    transform = transforms.Compose([
        transforms.Resize((28, 28)),  # Resize to 28x28 pixels
        transforms.ToTensor(),  # Convert to tensor and normalize to [0, 1]
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])

    # Apply the transformations
    tensor = transform(img)
    if tensor.mean() > 0:
        tensor = 1 - tensor

    # Ensure the tensor is of shape [1, 28, 28]
    tensor = tensor.squeeze()  # Remove any extra dimensions
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)  # Add channel dimension if needed

    plt.figure(figsize=(3, 3))
    plt.imshow(tensor.squeeze(), cmap='gray')
    plt.title("Converted MNIST-like Image")
    plt.axis('off')
    plt.savefig("transformed.png")

    # Save the image
    #dir_name, file_name = os.path.split(image_path)
    #name, ext = os.path.splitext(file_name)
    #new_file_name = f"{name}_mnist{ext}"
    #new_file_path = os.path.join(dir_name, new_file_name)
    #mnist_img = transforms.ToPILImage()(tensor)
    #mnist_img.save(new_file_path)
    #print(f"Saved converted image to: {new_file_path}")

    return tensor

# Example usage
if __name__ == "__main__":
    image_path = "data/six.png"
    mnist_tensor = png_to_mnist_tensor(image_path)
    print(f"Tensor shape: {mnist_tensor.shape}")
    print(f"Tensor value range: [{mnist_tensor.min():.2f}, {mnist_tensor.max():.2f}]")
