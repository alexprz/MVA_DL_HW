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
# vae.load_state_dict(state_dict)

os.makedirs('figs/q10/', exist_ok=True)
mnist = datasets.MNIST('MNIST/', train=False, download=True, transform=transforms.ToTensor())
loader = DataLoader(mnist, batch_size=64, shuffle=False)

# Ground truth
x, target = next(iter(loader))
imshow(make_grid(x))
plt.axis('off')
plt.savefig('figs/q10/mnist-ground-turth.pdf', bbox_inches='tight')

# Without training
x_reconstruct = vae.reconstruct(x)
imshow(make_grid(x_reconstruct))
plt.axis('off')
plt.savefig('figs/q10/mnist-vae-no-training.pdf', bbox_inches='tight')

# With training (10 epochs)
state_dict = torch.load('trained_models/vae-epochs_10-lr_0.001-bs_128-gclip_1.pth')
vae.load_state_dict(state_dict)

x_reconstruct = vae.reconstruct(x)
imshow(make_grid(x_reconstruct))
plt.axis('off')
plt.savefig('figs/q10/mnist-vae-10-epochs.pdf', bbox_inches='tight')

# With training (80 epochs)
state_dict = torch.load('trained_models/vae-epochs_80-lr_0.01-bs_128-gclip_1.pth')
vae.load_state_dict(state_dict)

x_reconstruct = vae.reconstruct(x)
imshow(make_grid(x_reconstruct))
plt.axis('off')
plt.savefig('figs/q10/mnist-vae-80-epochs.pdf', bbox_inches='tight')

plt.show()
