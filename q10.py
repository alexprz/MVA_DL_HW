"""Implement the question 10: plot reconstruction samples."""
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
# vae.load_state_dict(state_dict)


mnist = datasets.MNIST('MNIST/', train=True, download=True, transform=transforms.ToTensor())
loader = DataLoader(mnist, batch_size=64, shuffle=False)

# Ground truth
x, target = next(iter(loader))
imshow(make_grid(x))

# Without training
x_reconstruct = vae.reconstruct(x)
imshow(make_grid(x_reconstruct))

# With training (10 epochs)
state_dict = torch.load('trained_models/vae-epochs_10-lr_0.001-bs_128-gclip_1.pth')
vae.load_state_dict(state_dict)

x_reconstruct = vae.reconstruct(x)
imshow(make_grid(x_reconstruct))

plt.show()
