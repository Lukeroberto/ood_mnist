import torch
import torchvision as tv
from torch import nn
from torchvision import datasets, transforms
import torch.nn.functional as F

import matplotlib.pyplot as plt
from types import SimpleNamespace
import argparse

# modified from: https://github.com/udacity/deep-learning-v2-pytorch/blob/master/autoencoder/convolutional-autoencoder/Convolutional_Autoencoder_Solution.ipynb
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 1 --> 16), 3x3 kernels
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        # conv layer (depth from 16 --> 4), 3x3 kernels
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool2d(2, 2)

        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 1, 2, stride=2)

    def encode(self, x):
        ## encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # add second hidden layer
        x = F.relu(self.conv2(x))
        return self.pool(x)  # compressed representation
    
    def decode(self,x):
        ## decode ##
        # add transpose conv layers, with relu activation function
        x = F.relu(self.t_conv1(x))
        # output layer (with sigmoid for scaling from 0 to 1)
        return F.sigmoid(self.t_conv2(x))

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

def train(model, dataloader):
    for batch_idx, (data, target) in enumerate(dataloader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = distance(output, data)
        loss.backward()
        optimizer.step()
        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(dataloader.dataset),
                100. * batch_idx / len(dataloader), loss.item()))

def eval(autoencoder, n_samples=10, save_path="examples/reconstructions.png"):
    """
    Plots reconstructions of images taken from MNIST test dataset.
    """
    # Set up MNIST test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    mnist_test = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=n_samples, shuffle=True)

    # Get a batch of test examples
    images, _ = next(iter(test_loader))

    # Ensure autoencoder is in eval mode
    autoencoder.to("cpu")
    autoencoder.eval()

    with torch.no_grad():
        reconstructions = autoencoder(images)
        mse = distance(reconstructions, images)
        fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 2, 4))

        for i in range(n_samples):
            # Original images
            axes[0, i].imshow(images[i].squeeze().cpu(), cmap='gray')
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_title('Original')

            # Reconstructed images
            axes[1, i].imshow(reconstructions[i].squeeze().cpu(), cmap='gray')
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_title('Reconstructed')

        plt.suptitle(f'Autoencoder Reconstructions\nAverage MSE: {mse.item():.4f}', fontsize=16)
        plt.tight_layout()

        # Save the plot
        plt.savefig(save_path)
        plt.close(fig)

        print(f"Visualization saved to {save_path}")
        print(f"Average MSE: {mse.item():.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("epochs", type=int)
    cargs = parser.parse_args()

    # SimpleNamespaces are cool! Loose wrapper of dict that allow dot access
    args = SimpleNamespace(num_epochs=cargs.epochs, batch_size=64)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    kwargs = {'num_workers': 4, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    model = Autoencoder()
    model.to(device)
    distance = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(args.num_epochs):
        train(model, train_loader)
    eval(model)

    torch.save(model.state_dict(), "models/ood_detector.pt")
