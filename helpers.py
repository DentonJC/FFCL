#!/usr/bin/env python
# coding: utf-8

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL.ImageOps
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import confusion_matrix
from torchvision import datasets, transforms
from tqdm import tqdm
from torch.optim import Adam
import torch.nn.functional as F
from types import SimpleNamespace
from torchvision.datasets import MNIST, SVHN, CIFAR10, FashionMNIST
from avalanche.benchmarks import nc_benchmark
from os.path import expanduser
try:
    from avalanche.benchmarks.generators.benchmark_generators import benchmark_with_validation_stream
except:
    from avalanche.benchmarks.scenarios.validation_scenario import benchmark_with_validation_stream
from avalanche.benchmarks.classic import SplitMNIST, SplitCIFAR10, SplitFMNIST

def filter_memories(mem_images, mem_labels, task, device):
    indexes = np.where((mem_labels != task * 2) & (mem_labels != (task * 2) + 1))[0]
    return torch.from_numpy(mem_images[indexes]).to(device), torch.from_numpy(
        mem_labels[indexes]
    ).to(device)


    print(f"Random seed {seed} has been set.")


class Layer(nn.Linear):
    def __init__(self, in_features=None, out_features=None,
                 bias=True, device=None, dtype=None, lr=0.001, threshold = 2.0, num_epochs = 100, alpha=1e-4):
        super().__init__(in_features, out_features)
        self.relu = torch.nn.ReLU()
        self.opt = Adam(self.parameters(), lr=lr)
        self.threshold = threshold
        self.num_epochs = num_epochs
        self.alpha = alpha

    def forward(self, x):
        x_direction = x / (x.norm(2, 1, keepdim=True) + self.alpha)
        return self.relu(
            torch.mm(x_direction, self.weight.T) +
            self.bias.unsqueeze(0))

    def train(self, x_pos, x_neg):
        for i in range(self.num_epochs):
            g_pos = self.forward(x_pos).pow(2).mean(1)
            g_neg = self.forward(x_neg).pow(2).mean(1)
            # The following loss pushes pos (neg) samples to
            # values larger (smaller) than the self.threshold.
            loss = torch.log(1 + torch.exp(torch.cat([
                -g_pos + self.threshold,
                g_neg - self.threshold]))).mean()
            self.opt.zero_grad()
            # this backward just compute the derivative and hence
            # is not considered backpropagation.
            loss.backward()
            self.opt.step()
        return self.forward(x_pos).detach(), self.forward(x_neg).detach()

def overlay_y_on_x(x, y):
    """Replace the first 10 pixels of data [x] with one-hot-encoded label [y]
    """
    x_ = x.clone()
    x_[:, :10] *= 0.0
    #print('<<', x.shape, y.shape)
    x_[range(x.shape[0]), y] = x.max()
    return x_


class MLP_ff(nn.Module):
    def __init__(self, dims, device, lr=0.001, threshold = 2.0, num_epochs = 100, alpha=1e-4):
        super().__init__()
        self.layers = []
        for d in range(len(dims) - 1):
            self.layers += [Layer(in_features=dims[d], out_features=dims[d + 1], lr=lr, threshold=threshold, num_epochs=num_epochs, alpha=alpha).to(device)]

    def predict(self, x):
        goodness_per_label = []
        for label in range(10):
            h = overlay_y_on_x(x, label)
            goodness = []
            for layer in self.layers:
                h = layer(h)
                goodness += [h.pow(2).mean(1)]
            goodness_per_label += [sum(goodness).unsqueeze(1)]
        goodness_per_label = torch.cat(goodness_per_label, 1)
        return goodness_per_label.argmax(1)

    def train(self, x_pos, x_neg):
        h_pos, h_neg = x_pos, x_neg
        for i, layer in enumerate(self.layers):
            #print('training layer', i, '...')
            h_pos, h_neg = layer.train(h_pos, h_neg)

class MLP(nn.Module):
    def __init__(self, input_size=28 * 28, num_classes=10, nf=2000):
        super(MLP, self).__init__()

        self.input_size = np.prod(input_size)
        self.hidden = nn.Sequential(
            nn.Linear(self.input_size, nf),
            nn.ReLU(True),
            nn.Linear(nf, nf),
            nn.ReLU(True),
            nn.Linear(nf, nf),
            nn.ReLU(True),
            nn.Linear(nf, nf),
            nn.ReLU(True),
        )

        self.linear = nn.Linear(nf, num_classes)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        out = self.hidden(x)
        return self.linear(out)

def load_dataset(args, seed):    
    if args.dataset == "fmnist":
        classes = 10
        size = 28
        scenario = SplitFMNIST(5, fixed_class_order=[0,1,2,3,4,5,6,7,8,9], seed=seed)
    elif args.dataset == "cifar10":
        classes = 10
        size = 32
        scenario = SplitCIFAR10(5, fixed_class_order=[0,1,2,3,4,5,6,7,8,9], seed=seed)
    elif args.dataset == 'mnist':
        classes = 10
        size = 28
        scenario = SplitMNIST(5, fixed_class_order=[0,1,2,3,4,5,6,7,8,9], seed=seed)
    else:
        pass
    
    if args.valid:
        scenario = benchmark_with_validation_stream(scenario, shuffle=False, validation_size=0.2)
    
    params = {
        "scenario": scenario,
        "classes": classes,
        "size": size,
    }
    params = SimpleNamespace(**params)
    return params


def set_seed(seed=42):
    print('Setting seed:', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    print(f"Random seed {seed} has been set.")
