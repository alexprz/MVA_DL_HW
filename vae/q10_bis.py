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
# vae.load_state_dict(state_dict)

os.makedirs('figs/q10_bis/', exist_ok=True)
mnist = datasets.MNIST('MNIST/', train=True, download=True, transform=transforms.ToTensor())
loader = DataLoader(mnist, batch_size=2, shuffle=False)

# Ground truth
x, target = next(iter(loader))
# print(x.size())
x1, x2 = torch.split(x, split_size_or_sections=1, dim=0)
# print(x1.size())
# print(x2.size())
z1 = vae.encoder(x1)[0]
z2 = vae.encoder(x2)[0]
# print(z1.size())
# print(z2.size())
N = 8
Z = [alpha*z1 + (1-alpha)*z2 for alpha in np.linspace(0, 1, N)]
Z = torch.cat(Z, dim=0)

print(Z.shape)

X = vae.decoder(Z)

print(X.shape)

imshow(make_grid(X))
plt.axis('off')
plt.savefig('figs/q10_bis/mnist-vae.pdf', bbox_inches='tight')
plt.show()

exit()
imshow(make_grid(x))
plt.axis('off')
plt.savefig('figs/mnist-ground-turth.pdf', bbox_inches='tight')
plt.show()
exit()

# Without training
x_reconstruct = vae.reconstruct(x)
imshow(make_grid(x_reconstruct))
plt.axis('off')
plt.savefig('figs/mnist-vae-no-training.pdf', bbox_inches='tight')

# With training (10 epochs)
state_dict = torch.load('trained_models/vae-epochs_10-lr_0.001-bs_128-gclip_1.pth')
vae.load_state_dict(state_dict)

x_reconstruct = vae.reconstruct(x)
imshow(make_grid(x_reconstruct))
plt.axis('off')
plt.savefig('figs/mnist-vae-10-epochs.pdf', bbox_inches='tight')

# With training (80 epochs)
state_dict = torch.load('trained_models/vae-epochs_80-lr_0.01-bs_128-gclip_1.pth')
vae.load_state_dict(state_dict)

x_reconstruct = vae.reconstruct(x)
imshow(make_grid(x_reconstruct))
plt.axis('off')
plt.savefig('figs/mnist-vae-80-epochs.pdf', bbox_inches='tight')

plt.show()
