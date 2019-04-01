import sys
sys.path.append('../pyvacy')
sys.path.append('../competitor_pack')

import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.optim import Adam

from pyvacy.optimizers.dp_optimizer import DPAdam
from pyvacy.analysis import privacy_accountant
from competitor_pack.dataset import ColoradoDataset
from competitor_pack.postprocess import ColoradoDatasetPostprocessor

print(privacy_accountant.epsilon(N=60000, batch_size=256, noise_multiplier=1.1, epochs=60, delta=1e-5))

os.makedirs('samples', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--input_size', type=int, default=735, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=400, help='interval betwen image samples')
parser.add_argument('--l2_norm_clip', type=float, default=0.75, help='')
parser.add_argument('--noise_multiplier', type=float, default=0.3, help='')
parser.add_argument('--epsilon', type=float, default=1.0, help='')
opt = parser.parse_args()
print(opt)

input_size = opt.input_size

cuda = True if torch.cuda.is_available() else False

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, input_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Configure data loader
os.makedirs('data/mnist', exist_ok=True)
dataloader = torch.utils.data.DataLoader(ColoradoDataset(), batch_size=opt.batch_size, shuffle=True)
postprocessor = ColoradoDatasetPostprocessor()

# Optimizers
optimizer_G = Adam(params=generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = DPAdam(l2_norm_clip=opt.l2_norm_clip, noise_multiplier=opt.noise_multiplier, num_minibatches=1, params=discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

for epoch in range(1, 10):
    print('epochs: {}, eps: {}'.format(epoch, privacy_accountant.epsilon(len(dataloader), opt.batch_size, opt.noise_multiplier, epoch, 1/(len(dataloader)**2))))

# ----------
#  Training
# ----------

for epoch in range(1, opt.epochs+1):
    for i, samples in enumerate(dataloader):
        # Adversarial ground truths
        valid = Variable(Tensor(samples.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(samples.size(0), 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_samples = Variable(samples.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (samples.shape[0], opt.latent_dim))))

        # Generate a batch of images
        gen_samples = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_samples), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_samples), valid)
        fake_loss = adversarial_loss(discriminator(gen_samples.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, opt.epochs, i, len(dataloader),
                                                            d_loss.item(), g_loss.item()))

    gen_samples = generator(Variable(Tensor(np.random.normal(0, 1, (100000, opt.latent_dim))))) # should be 661967
    if epoch in [1, 5, 10, 15, 25, 50, 100, 200]:
        torch.save(generator, 'samples/model_epoch={}.pt'.format(epoch))
        torch.save(gen_samples, 'samples/raw_samples_epoch={}.pt'.format(epoch))
        postprocessor.postprocess('samples/raw_samples_epoch={}.pt'.format(epoch), 'samples/samples_epoch={}.csv'.format(epoch))
