import torch
import torchvision as tv
from torch import nn
from torchvision import datasets, transforms

from types import SimpleNamespace

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder,self).__init__()
        self.encoder = nn.Sequential(
            # 28 x 28
            nn.Conv2d(1, 4, kernel_size=5),
            # 4 x 24 x 24
            nn.ReLU(True),
            nn.Conv2d(4, 8, kernel_size=5),
            nn.ReLU(True),
            # 8 x 20 x 20 = 3200
            nn.Flatten(),
            nn.Linear(3200, 10),
            # 10
            nn.Softmax(),
            )
        self.decoder = nn.Sequential(
            # 10
            nn.Linear(10, 400),
            # 400
            nn.ReLU(True),
            nn.Linear(400, 4000),
            # 4000
            nn.ReLU(True),
            nn.Unflatten(1, (10, 20, 20)),
            # 10 x 20 x 20
            nn.ConvTranspose2d(10, 10, kernel_size=5),
            # 24 x 24
            nn.ConvTranspose2d(10, 1, kernel_size=5),
            # 28 x 28
            nn.Sigmoid(),
            )
    def forward(self, x):
        enc = self.encoder(x)
        dec = self.decoder(enc)
        return dec

def train(model, dataloader):
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = distance(output, data)
        loss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(dataloader.dataset),
                100. * batch_idx / len(dataloader), loss.item()))


if __name__ == "__main__":

    args = SimpleNamespace(num_epochs=5, batch_size=256)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    kwargs = {'num_workers': 4, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    model = Autoencoder()
    model.to(device)
    distance = nn.MSELoss()
    #distance = nn.BCELoss()
    #distance = nn.BCEWithLogitsLoss()
    #optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    for epoch in range(args.num_epochs):
        # Train
        train(model, train_loader)

    torch.save(model, "ood_detector.pt")
