"""Implement the question 10: plot reconstruction samples."""
import os
import torch
from torchvision.utils import make_grid
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import numpy as np

from vae import VAE


def imshow(img):
    plt.figure()
    plt.imshow(np.transpose(img.detach().numpy(), (1, 2, 0)))


vae = VAE()
state_dict = torch.load('trained_models/vae-epochs_80-lr_0.01-bs_128-gclip_1.pth')
vae.load_state_dict(state_dict)

os.makedirs('figs/q10_bis/', exist_ok=True)
mnist = datasets.MNIST('MNIST/', train=True, download=True, transform=transforms.ToTensor())
loader = DataLoader(mnist, batch_size=2, shuffle=False)

iterator = iter(loader)
Xs = []

N = 16  # Number of samples in the linear interpolation
M = 8  # Number of interpolations
for _ in range(M):
    x, target = next(iterator)
    x1, x2 = torch.split(x, split_size_or_sections=1, dim=0)

    z1 = vae.encoder(x1)[0]
    z2 = vae.encoder(x2)[0]

    Z = [alpha*z1 + (1-alpha)*z2 for alpha in np.linspace(0, 1, N)]
    Z = torch.cat(Z, dim=0)

    X = vae.decoder(Z)

    Xs.append(X)

Xs = torch.cat(Xs, dim=0)

imshow(make_grid(Xs, nrow=N))
plt.axis('off')
plt.savefig(f'figs/q10_bis/mnist-vae-N_{N}-M_{M}.pdf', bbox_inches='tight')
plt.show()
