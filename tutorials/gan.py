import sys
sys.path.append('../pyvacy')

import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from pyvacy import optim, analysis, sampling


# Deterministic output
torch.manual_seed(0)
np.random.seed(0)


class Generator(nn.Module):
    def __init__(self, input_dim, output_dim, device='cpu'):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.LayerNorm(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(input_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, output_dim),
            nn.Tanh()
        ).to(device)

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], 1, 28, 28)
        return img


class Discriminator(nn.Module):
    def __init__(self, input_dim, device='cpu'):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        ).to(device)

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity


def train(params):
    train_dataset = datasets.MNIST('data/mnist',
        train=True,
        download=True,
        transform=transforms.Compose([
           transforms.ToTensor(),
           transforms.Normalize((0.5,), (0.5,))
        ])
    )

    generator = Generator(
        input_dim=params['latent_dim'],
        output_dim=np.prod(train_dataset[0][0].shape),
        device=params['device'],
    )

    g_optimizer = torch.optim.RMSprop(
        params=generator.parameters(),
        lr=params['lr'],
        weight_decay=params['l2_penalty'],
    )

    discriminator = Discriminator(
        input_dim=np.prod(train_dataset[0][0].shape),
        device=params['device']
    )

    d_optimizer = optim.DPRMSprop(
        l2_norm_clip=params['l2_norm_clip'],
        noise_multiplier=params['noise_multiplier'],
        minibatch_size=params['minibatch_size'],
        microbatch_size=params['microbatch_size'],
        params=discriminator.parameters(),
        lr=params['lr'],
        weight_decay=params['l2_penalty'],
    )

    print('Achieves ({}, {})-DP'.format(
        analysis.epsilon(
            len(train_dataset),
            params['minibatch_size'],
            params['noise_multiplier'],
            params['iterations'],
            params['delta']
        ),
        params['delta'],
    ))

    minibatch_loader, microbatch_loader = sampling.get_data_loaders(
        params['minibatch_size'],
        params['microbatch_size'],
        params['iterations']
    )

    for iteration, (X_minibatch, _) in enumerate(minibatch_loader(train_dataset)):
        d_optimizer.zero_grad()
        for X_microbatch in microbatch_loader(X_minibatch):
            X_microbatch = X_microbatch.to(params['device'])

            z = torch.randn(X_microbatch.size(0), params['latent_dim'], device=params['device'])
            fake = generator(z).detach()
            d_optimizer.zero_microbatch_grad()
            d_loss = -torch.mean(discriminator(X_microbatch)) + torch.mean(discriminator(fake))
            d_loss.backward()
            d_optimizer.microbatch_step()
        d_optimizer.step()

        for parameter in discriminator.parameters():
            parameter.data.clamp_(-params['clip_value'], params['clip_value'])

        if iteration % params['d_updates'] == 0:
            z = torch.randn(X_minibatch.size(0), params['latent_dim'], device=params['device'])
            fake = generator(z)
            g_optimizer.zero_grad()
            g_loss = -torch.mean(discriminator(fake))
            g_loss.backward()
            g_optimizer.step()

        if iteration % 100 == 0:
            print('[Iteration %d/%d] [D loss: %f] [G loss: %f]' % (iteration, params['iterations'], d_loss.item(), g_loss.item()))
            z = torch.randn(X_minibatch.size(0), params['latent_dim'], device=params['device'])
            fake = generator(z)
            save_image(fake.data[:25], "%d.png" % iteration, nrow=5, normalize=True)

    return generator


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip-value', type=float, default=0.01, help='upper bound on weights of the discriminator (default: 0.01)')
    parser.add_argument('--d-updates', type=int, default=5, help='number of iterations to update discriminator per generator update (default: 5)')
    parser.add_argument('--delta', type=float, default=1e-5, help='delta for epsilon calculation (default: 1e-5)')
    parser.add_argument('--device', type=str, default=('cuda' if torch.cuda.is_available() else 'cpu'), help='whether or not to use cuda (default: cuda if available)')
    parser.add_argument('--iterations', type=int, default=30000, help='number of iterations to train (default: 30000)')
    parser.add_argument('--l2-norm-clip', type=float, default=0.35, help='upper bound on the l2 norm of gradient updates (default: 0.35)')
    parser.add_argument('--l2-penalty', type=float, default=0., help='l2 penalty on model weights (default: 0.)')
    parser.add_argument('--latent-dim', type=int, default=128, help='dimensionality of the latent space (default: 128)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 5e-5)')
    parser.add_argument('--microbatch-size', type=int, default=1, help='input microbatch size for training (default: 1)')
    parser.add_argument('--minibatch-size', type=int, default=128, help='input minibatch size for training (default: 64)')
    parser.add_argument('--noise-multiplier', type=float, default=1.1, help='ratio between clipping bound and std of noise applied to gradients (default: 1.1)')
    params = vars(parser.parse_args())

    generator = train(params)

    with open('dp_generator.dat', 'wb') as f:
        torch.save(generator, f)
