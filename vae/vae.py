"""Implement the VAE example of the homework (questions 9, 10)."""
import os
import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import numpy as np

from train import train_vae


class VAE(nn.Module):
    """Implement the Variational Auto Encoder."""

    def __init__(self, input_size=28*28, n_hidden_neurons=200, n_latent_dim=20):
        """Init.

        Args:
        -----
            input_size : int
            n_hidden_neurons : int

        """
        super().__init__()
        self.input_size = input_size

        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # Encoder's layers
        self.e_fc1 = nn.Linear(input_size, n_hidden_neurons)
        self.e_fc2_mu = nn.Linear(n_hidden_neurons, n_latent_dim)
        self.e_fc2_logvar = nn.Linear(n_hidden_neurons, n_latent_dim)

        # Decoder's layers
        self.d_fc3 = nn.Linear(n_latent_dim, n_hidden_neurons)
        self.d_fc4 = nn.Linear(n_hidden_neurons, input_size)
        self.decode = nn.Sequential(
            self.d_fc3,
            self.relu,
            self.d_fc4,
            self.sigmoid,
        )

    def encoder(self, x):
        """Implement the encoder part of the network."""
        x = x.view(-1, self.input_size)
        x = self.relu(self.e_fc1(x))
        mu = self.e_fc2_mu(x)
        logvar = self.e_fc2_logvar(x)

        return mu, logvar

    def decoder(self, z):
        """Implement the decoder part of the network."""
        x = self.decode(z)
        s = np.sqrt(self.input_size).astype(int)
        x = x.view(-1, 1, s, s)
        return x

    def forward(self, x):
        """Compute an ouput given an input."""
        # Encode x
        mu, logvar = self.encoder(x)

        # Sample a z in the latent space
        std = torch.exp(.5*logvar)
        eps = torch.randn_like(logvar)
        z = mu + eps*std

        # Decode the sampled z
        x = self.decoder(z)

        # Return. Note that we also return mu and logvar to compute the
        # regularization term of the loss
        return x, mu, logvar

    def reconstruct(self, x):
        x_reconstruced = self.forward(x)[0]
        s = np.sqrt(self.input_size).astype(int)
        return x_reconstruced.view(-1, 1, s, s)


if __name__ == '__main__':
    # Create the VAE model
    vae = VAE(input_size=28*28, n_hidden_neurons=200, n_latent_dim=20)

    # Load the MNIST dataset
    mnist = datasets.MNIST('MNIST/', train=True, download=True, transform=transforms.ToTensor())

    # Training parameters
    batch_size = 128
    lr = 0.01
    n_epochs = 80
    gclip = 1

    loader = DataLoader(mnist, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.SGD(params=vae.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    # sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr,
    #                                             epochs=n_epochs,
    #                                             steps_per_epoch=len(loader),
    #                                             )

    for epoch in range(1, n_epochs+1):
        train_vae(vae, loader, optimizer, epoch=epoch, gradient_clip=gclip)
        sched.step()

    os.makedirs('trained_models/', exist_ok=True)
    torch.save(vae.state_dict(), f'trained_models/vae-epochs_{n_epochs}-lr_{lr}-bs_{batch_size}-gclip_{gclip}.pth')
