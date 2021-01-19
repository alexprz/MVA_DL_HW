"""Implement the VAE example of the homework (questions 9, 10, 11)."""
import torch
import torch.nn as nn


class VAE(nn.Module):
    """Implement the Variational Auto Encoder."""

    def __init__(self, input_size, n_hidden_neurons, n_latent_dim=20):
        """Init.

        Args:
        -----
            input_size : int
            n_hidden_neurons : int

        """
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
            self.sigmoid
        )

    def encoder(self, x):
        """Implement the encoder part of the network."""
        x = self.relu(self.e_fc1(x))
        mu = self.e_fc2_mu(x)
        logvar = self.e_fc2_logvar(x)

        return mu, logvar

    def decoder(self, z):
        """Implement the decoder part of the network."""
        return self.decode(z)

    def forward(self, x):
        """Compute an ouput given an input."""
        # Encode x
        x = x.view(1, self.input_size)
        mu, logvar = self.encode(x)

        # Sample a z in the latent space
        std = torch.exp(.5*logvar)
        eps = torch.randn_like(logvar)
        z = mu + eps*std

        # Decode the sampled z
        return self.decode(z)
