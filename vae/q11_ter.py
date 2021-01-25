"""Implement the question 11."""
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

os.makedirs('figs/q11_v2/', exist_ok=True)

# Sample in the latent space following N(0, 1)
Z = torch.randn((64, 20))

# Without training
x_reconstruct = vae.decoder(Z)
imshow(make_grid(x_reconstruct))
plt.axis('off')
plt.savefig('figs/q11_v2/mnist-vae-no-training.pdf', bbox_inches='tight')

# With training (10 epochs)
state_dict = torch.load('trained_models/unet-vae-v4/unet-vae-epochs_10-lr_0.0001-bs_128-gclip_1.pth', map_location=torch.device('cpu'))
vae.load_state_dict(state_dict)

x_reconstruct = vae.decoder(Z)
imshow(make_grid(x_reconstruct))
plt.axis('off')
plt.savefig('figs/q11_v2/mnist-vae-10-epochs.pdf', bbox_inches='tight')

# With training (80 epochs)
state_dict = torch.load('trained_models/unet-vae-v4/unet-vae-epochs_80-lr_0.0001-bs_128-gclip_1.pth', map_location=torch.device('cpu'))
vae.load_state_dict(state_dict)

x_reconstruct = vae.decoder(Z)
imshow(make_grid(x_reconstruct))
plt.axis('off')
plt.savefig('figs/q11_v2/mnist-vae-80-epochs.pdf', bbox_inches='tight')

plt.show()
