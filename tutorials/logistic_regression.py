import sys
sys.path.append('../pyvacy')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from pyvacy import optim, analysis, sampling

# Deterministic output
torch.manual_seed(0)
np.random.seed(0)

# Fake dataset where label is 1 iff the number of 1's
# in the feature vector is at least 2.
train_dataset = TensorDataset(
    torch.Tensor(100 * [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ]),
    torch.Tensor(100 * [
        [0],
        [0],
        [0],
        [1],
        [0],
        [1],
        [1],
        [1],
    ])
)

# Training parameters.
N = len(train_dataset)
minibatch_size = 16
microbatch_size = 1
iterations = 200
l2_norm_clip = 0.7
noise_multiplier = 1.
delta = 1e-5
lr = 0.15

model = nn.Sequential(
    nn.Linear(len(next(iter(train_dataset))[0]), len(next(iter(train_dataset))[1])),
    nn.Sigmoid()
)

loss_function = torch.nn.BCELoss()

optimizer = optim.DPSGD(
    l2_norm_clip=l2_norm_clip,
    noise_multiplier=noise_multiplier,
    minibatch_size=minibatch_size,
    microbatch_size=microbatch_size,
    params=model.parameters(),
    lr=lr,
)

minibatch_loader, microbatch_loader = sampling.get_data_loaders(
    minibatch_size,
    microbatch_size,
    iterations
)

print('Achieves ({}, {})-DP'.format(
    analysis.epsilon(
        len(train_dataset),
        minibatch_size,
        noise_multiplier,
        iterations,
        delta,
    ),
    delta,
))

for X_minibatch, y_minibatch in minibatch_loader(train_dataset):
    optimizer.zero_grad()
    for X_microbatch, y_microbatch in microbatch_loader(TensorDataset(X_minibatch, y_minibatch)):
        optimizer.zero_microbatch_grad()
        y_pred = model(X_microbatch)
        loss = loss_function(y_pred, y_microbatch)
        loss.backward()
        optimizer.microbatch_step()
    optimizer.step()

print(list(model.parameters()))

