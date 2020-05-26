import sys
sys.path.append('../pyvacy')

import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms

from pyvacy import optim, analysis, sampling


# Deterministic output
torch.manual_seed(0)
np.random.seed(0)

class Flatten(nn.Module):
    def forward(self, inp):
        return inp.reshape(inp.shape[0], -1)

class Classifier(nn.Module):
    def __init__(self, input_dim, device='cpu'):
        super(Classifier, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, 8, 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 1),
            nn.Conv2d(16, 32, 4, 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 1),
            Flatten(),
            nn.Linear(288, 10),
            nn.LogSoftmax(dim=1)
        ).to(device)

    def forward(self, x):
        return self.model(x)


def train(params):
    train_dataset = datasets.MNIST('data/mnist',
        train=True,
        download=True,
        transform=transforms.Compose([
           transforms.ToTensor(),
           transforms.Normalize((0.5,), (0.5,))
        ])
    )

    classifier = Classifier(
        input_dim=np.prod(train_dataset[0][0].shape),
        device=params['device']
    )

    optimizer = optim.DPSGD(
        l2_norm_clip=params['l2_norm_clip'],
        noise_multiplier=params['noise_multiplier'],
        minibatch_size=params['minibatch_size'],
        microbatch_size=params['microbatch_size'],
        params=classifier.parameters(),
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

    loss_function = nn.NLLLoss()

    minibatch_loader, microbatch_loader = sampling.get_data_loaders(
        params['minibatch_size'],
        params['microbatch_size'],
        params['iterations']
    )

    iteration = 0
    for X_minibatch, y_minibatch in minibatch_loader(train_dataset):
        optimizer.zero_grad()
        for X_microbatch, y_microbatch in microbatch_loader(TensorDataset(X_minibatch, y_minibatch)):
            X_microbatch = X_microbatch.to(params['device'])
            y_microbatch = y_microbatch.to(params['device'])

            optimizer.zero_microbatch_grad()
            loss = loss_function(classifier(X_microbatch), y_microbatch)
            loss.backward()
            optimizer.microbatch_step()
        optimizer.step()

        if iteration % 10 == 0:
            print('[Iteration %d/%d] [Loss: %f]' % (iteration, params['iterations'], loss.item()))
        iteration += 1

    return classifier


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--delta', type=float, default=1e-5, help='delta for epsilon calculation (default: 1e-5)')
    parser.add_argument('--device', type=str, default=('cuda' if torch.cuda.is_available() else 'cpu'), help='whether or not to use cuda (default: cuda if available)')
    parser.add_argument('--iterations', type=int, default=14000, help='number of iterations to train (default: 14000)')
    parser.add_argument('--l2-norm-clip', type=float, default=1., help='upper bound on the l2 norm of gradient updates (default: 0.1)')
    parser.add_argument('--l2-penalty', type=float, default=0.001, help='l2 penalty on model weights (default: 0.001)')
    parser.add_argument('--lr', type=float, default=0.15, help='learning rate (default: 0.15)')
    parser.add_argument('--microbatch-size', type=int, default=1, help='input microbatch size for training (default: 1)')
    parser.add_argument('--minibatch-size', type=int, default=256, help='input minibatch size for training (default: 256)')
    parser.add_argument('--noise-multiplier', type=float, default=1.1, help='ratio between clipping bound and std of noise applied to gradients (default: 1.1)')
    params = vars(parser.parse_args())

    classifier = train(params)

    with open('dp_classifier.dat', 'wb') as f:
        torch.save(classifier, f)

    test_dataset = datasets.MNIST('data/mnist',
        train=False,
        download=True,
        transform=transforms.Compose([
           transforms.ToTensor(),
           transforms.Normalize((0.5,), (0.5,))
        ])
    )

    X, y = next(iter(DataLoader(test_dataset, batch_size=len(test_dataset))))
    X, y  = X.to('cuda'), y.to('cuda')

    y_pred = classifier(X).max(1)[1]

    count = 0.
    correct = 0.
    for pred, actual in zip(y_pred, y):
        if pred.item() == actual.item():
            correct += 1.
        count += 1.
    print('Test Accuracy: {}'.format(correct / count))

