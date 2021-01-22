"""Implement the UNet VAE example of the homework (questions 11)."""
import os
import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import numpy as np

from train import train_vae


class UNetVAE(nn.Module):


    def __init__(self):
        """Init."""

        super().__init__()

        self.encode = nn.Sequential(  # 28, 28
            nn.Conv2d(1, 10, kernel_size=3),  # 26, 26
            nn.ReLU(),
            nn.MaxPool2d(2),  # 13, 13
            nn.Conv2d(10, 20, kernel_size=3),  # 11, 11
            nn.ReLU(),
            nn.MaxPool2d(2),  # 6, 6
            nn.Conv2d(20, 40, kernel_size=3),  # 4, 4
            nn.ReLU(),
            nn.MaxPool2d(2),  # 2, 2
            # nn.Conv2d(30, 40, kernel_size=3),  # 1, 1
            # nn.ReLU(),
        )

        self.decode = nn.Sequential(  # 1, 1
            nn.Upsample(scale_factor=12),  # 12, 12
            nn.Conv2d(20, 15, kernel_size=3),  # 10, 10
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # 20, 20
            nn.Conv2d(15, 10, kernel_size=5),  # 16, 16
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # 32, 32
            nn.Conv2d(10, 1, kernel_size=5),  # 28, 28
            nn.Sigmoid(),
        )

    def encoder(self, x):
        """Implement the encoder part of the network."""
        x = self.encode(x)  # 1, 1
        mu, logvar = torch.split(x, split_size_or_sections=20, dim=1)

        # print(mu.size())
        # print(logvar.size())
        # exit()

        mu = mu.view(-1, 20)  # 20
        logvar = logvar.view(-1, 20)  # 20

        return mu, logvar

    def decoder(self, z):
        """Implement the decoder part of the network."""
        return self.decode(z)

    def forward(self, x):
        """Compute an ouput given an input."""
        # Encode x
        mu, logvar = self.encoder(x)


        # Sample a z in the latent space
        std = torch.exp(.5*logvar)
        eps = torch.randn_like(logvar)
        z = mu + eps*std

        # print(mu.size())
        # print(logvar.size())
        # print(z.size())
        z = z.view(-1, 20, 1, 1)
        # print(z.size())
        # exit()
        # Decode the sampled z
        x = self.decoder(z)

        # Return. Note that we also return mu and logvar to compute the
        # regularization term of the loss
        return x, mu, logvar


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    if use_cuda :
        device = torch.device("cuda")
        print("using GPU")
    else :
        device = torch.device("cpu")
        print("using CPU")

    # Create the VAE model
    unet_vae = UNetVAE().to(device)

    # Load the MNIST dataset
    mnist = datasets.MNIST('MNIST/', train=True, download=True, transform=transforms.ToTensor())

    # Training parameters
    batch_size = 128
    lr = 0.001
    n_epochs = 80
    gclip = 1
    save_points = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80]

    loader = DataLoader(mnist, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.SGD(params=unet_vae.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    # sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr,
    #                                             epochs=n_epochs,
    #                                             steps_per_epoch=len(loader),
    #                                             )

    os.makedirs('trained_models/unet-vae-v2/', exist_ok=True)

    for epoch in range(1, n_epochs+1):
        train_vae(unet_vae, loader, optimizer, epoch=epoch, gradient_clip=gclip, log_interval=100, use_cuda=use_cuda)
        sched.step()

        if epoch in save_points:
            print('Saving model')
            torch.save(unet_vae.state_dict(), f'trained_models/unet-vae-v2/unet-vae-epochs_{epoch}-lr_{lr}-bs_{batch_size}-gclip_{gclip}.pth')

    torch.save(unet_vae.state_dict(), f'trained_models/unet-vae-v2/unet-vae-epochs_{n_epochs}-lr_{lr}-bs_{batch_size}-gclip_{gclip}.pth')

