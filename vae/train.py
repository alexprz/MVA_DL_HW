"""Implement the training procedure for the models."""
import torch
from torch.nn import functional as F


def train_vae(model, loader, optimizer, log_interval=100,
              use_cuda=False, epoch=None, gradient_clip=None):
    """Train the model, one call for each epoch."""
    model.train()

    for batch_idx, (x, target) in enumerate(loader):
        if use_cuda:
            x, target = x.cuda(), target.cuda()

        optimizer.zero_grad()

        x_reconstruct, mu, logvar = model(x)
        x = x.view(-1, model.input_size)

        # Compute the reconstruction loss
        # print('recons', x_reconstruct)
        # print('x', x)
        reconstruction_loss = F.binary_cross_entropy(x_reconstruct, x, reduction='sum')

        # Compute the regularization loss. Here we have a closed form
        # according to https://arxiv.org/abs/1312.6114 appendix B
        regularization_loss = -.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        loss = reconstruction_loss + regularization_loss
        loss.backward()

        if gradient_clip:
            torch.nn.utils.clip_grad_value_(model.parameters(),
                                            clip_value=gradient_clip)

        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch {}: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(x), len(loader.dataset),
                100. * batch_idx / len(loader), loss.data.item()))
