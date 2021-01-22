"""Implement the question 10: plot reconstruction samples."""
import os
import torch
from torchvision.utils import make_grid
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import numpy as np

from unet_vae import UNetVAE


def imshow(img):
    plt.figure()
    plt.imshow(np.transpose(img.detach().numpy(), (1, 2, 0)))



vae = UNetVAE()
# vae.load_state_dict(state_dict)

os.makedirs('figs/', exist_ok=True)
mnist = datasets.MNIST('MNIST/', train=True, download=True, transform=transforms.ToTensor())
loader = DataLoader(mnist, batch_size=64, shuffle=False)

# Ground truth
x, target = next(iter(loader))
imshow(make_grid(x))
plt.axis('off')
plt.savefig('figs/mnist-ground-turth.pdf', bbox_inches='tight')

# Without training
x_reconstruct = vae.forward(x)[0]
imshow(make_grid(x_reconstruct))
plt.axis('off')
plt.savefig('figs/mnist-vae-no-training.pdf', bbox_inches='tight')

# With training (10 epochs)
state_dict = torch.load('trained_models/unet-vae-epochs_20-lr_0.0001-bs_128-gclip_1.pth', map_location=torch.device('cpu'))
vae.load_state_dict(state_dict)

x_reconstruct = vae.forward(x)[0]
imshow(make_grid(x_reconstruct))
plt.axis('off')
plt.savefig('figs/mnist-vae-10-epochs.pdf', bbox_inches='tight')

# With training (80 epochs)
# state_dict = torch.load('trained_models/vae-epochs_80-lr_0.01-bs_128-gclip_1.pth', map_location=torch.device('cpu'))
# vae.load_state_dict(state_dict)

# x_reconstruct = vae.forward(x)[0]
# imshow(make_grid(x_reconstruct))
# plt.axis('off')
# plt.savefig('figs/mnist-vae-80-epochs.pdf', bbox_inches='tight')

plt.show()
