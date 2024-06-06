#!/usr/bin/env python
# coding: utf-8

import argparse
from pprint import pprint
from types import SimpleNamespace
import sys
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from continuum import ClassIncremental, rehearsal
from continuum.datasets import SVHN, CIFAR10, MNIST, FashionMNIST
from continuum import Permutations, Rotations
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import transforms
from tqdm import tqdm
import random

from helpers import MLP_ff, MLP, overlay_y_on_x, set_seed, load_dataset


def tensor_element_swapper(target, device):
    unique_elements = torch.unique(target)

    # Create a tensor to store the new values
    new_target = torch.zeros_like(target).to(device)
    
    # Generate random indices, avoiding self-replacement
    for unique_element in unique_elements:
        mask = (target == unique_element)
        choices = [el for el in unique_elements if el != unique_element]
        random_values = torch.tensor(random.choices(choices, k=mask.sum())).to(device)
        new_target[mask] = random_values

    return new_target


def train(
    args,
    model,
    device,
    train_loader,
    optimizer,
    criterion,
    memory,
    task,
):
    if args.model != 'mlp_ff':
        model.train()
    for batch_idx, (data, target, tasks) in tqdm(enumerate(train_loader)):
        #print('mew')
        if task != 0:
            # TODO: extract without task
            mem_images, mem_labels, _ = memory.get()
            memory.add(
                data.detach().cpu(),
                target.detach().cpu(),
                tasks.detach().cpu(),
                None,
            )

            indexes = np.where(
                (mem_labels != task * 2) & (mem_labels != (task * 2) + 1)
            )[0]
            mem_images = torch.from_numpy(mem_images[indexes]).to(device)
            mem_labels = torch.from_numpy(mem_labels[indexes]).to(device)

            if args.batch_size_mem <= mem_images.shape[0]:
                indices = np.random.choice(
                    mem_images.shape[0], size=args.batch_size_mem, replace=False
                )
            else:
                indices = list(range(mem_images.shape[0]))
            mem_images = mem_images[indices]
            mem_labels = mem_labels[indices]

            data = torch.cat([data.to(device), mem_images], axis=0)
            target = torch.cat([target.to(device), mem_labels], axis=0)
        else:
            memory.add(
                data.detach().cpu(),
                target.detach().cpu(),
                tasks.detach().cpu(),
                None,
            )

        data, target = data.to(device), target.to(device)

        if args.model == 'mlp':
            for _ in range(args.epochs):
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            #print('Not implemented!')
        else:
            #unique_elements = torch.unique(target)
            #num_unique = len(unique_elements)
            
            x_pos = overlay_y_on_x(data.reshape(len(data), -1), target)
            rnd = torch.randperm(data.size(0))
            x_neg = overlay_y_on_x(data.reshape(len(data), -1), target[rnd])

            model.train(x_pos, x_neg)


def test(model, loader, device):
    correct = 0.0
    total = 0.0

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            labels = labels.to(device)

            pred = model.predict(images.reshape(len(images), -1))
            total += labels.size(0)
            correct += (pred == labels.data).sum().cpu().numpy()

    return correct / total

def test_baseline(model, loader, device):
    model.eval()
    correct = 0.0
    total = 0.0

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            pred = model(images)
            pred = torch.max(pred.data, 1)[1].cpu()
            correct += (pred == labels.data).sum().numpy()
            total += labels.size(0)

    return correct / total


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        "-d",
        default="fmnist",
        choices=["cifar10", "svhn", "fmnist", "mnist"],
    )
    parser.add_argument(
        "--model", default="mlp_ff", choices=["mlp_ff", "mlp"]
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="input batch size for training (default: 10)",
    )
    parser.add_argument(
        "--batch_size_mem",
        type=int,
        default=256,
        help="mem batch size for training (default: 10)",
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="number of epochs to train (default: 1)"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.03, help="learning rate"
    )
    # parser.add_argument("--weight_decay", type=float, default=0.0005, help="weight_decay")
    parser.add_argument("--trials", type=int, default=5, help="trials")
    parser.add_argument("--memory_size", type=int, default=500, help="memory_size")
    parser.add_argument("--alpha", type=float, default=0.04)
    parser.add_argument("--threshold", type=float, default=2)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--valid", action="store_true")
    parser.set_defaults(valid=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    if args.batch_size_mem < 0:
        args.batch_size_mem = args.memory_size
    writer = SummaryWriter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pprint(args)

    totals = []
    for iteration in range(args.trials):
        print("Iteration:", iteration)
        set_seed(seed=iteration)
        params = load_dataset(args, iteration)

        if args.model == "mlp":
            model = MLP(num_classes=params.classes, input_size=28*28).to(device)
            optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
        else:            
            model = MLP_ff(
                [28*28, 2000, 2000, 2000, 2000],
                device,
                args.learning_rate,
                args.threshold,
                args.num_epochs,
                args.alpha,
            )#.to(device)
            optimizer = None

        criterion = nn.CrossEntropyLoss()
        
        memory = rehearsal.RehearsalMemory(
            memory_size=args.memory_size, herding_method="random", fixed_memory=False
        )
        
        if args.valid:
            val_data = params.scenario.valid_stream
        else:
            val_data = params.scenario.test_stream
        for task in tqdm(range(len(params.scenario.train_stream))):
            train_data = params.scenario.train_stream[task]
            
            train_loader = DataLoader(
                train_data.dataset,
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=True,
                num_workers=2,
            )
            train(
                args,
                model,
                device,
                train_loader,
                optimizer,
                criterion,
                memory,
                task,
            )

            val_accs = []
            for j in range(task + 1):
                test_loader = DataLoader(
                    val_data[j].dataset,
                    batch_size=512,
                    shuffle=False,
                    pin_memory=True,
                    num_workers=2,
                )
                
                if args.model == 'mlp_ff':
                    val_acc = test(model, test_loader, device)
                else:
                    val_acc = test_baseline(model, test_loader, device)
                val_accs += [val_acc]
            print(np.array(val_accs), np.mean(val_accs))

            """
            if task == 0:
                if np.mean(val_accs) < 0.8:
                    writer.add_scalar("accuracy", -1)
                    sys.exit(0)
            if task >= 3:
                if np.mean(val_accs) < 0.7:
                    writer.add_scalar("accuracy", -1)
                    sys.exit(0)
            """

        totals += [np.mean(val_accs)]
        print("accuracy:", round(np.mean(totals) * 100, 2))
        print("std:", round(np.std(totals) * 100, 2))
        print("accs:", str(totals))
        writer.add_scalar("iter_acc", round(np.mean(totals) * 100, 2))
        writer.add_scalar("std", round(np.std(totals) * 100, 2))
        

    if args.memory_size == 200:
        threshold = 0.7
    elif args.memory_size == 500:
        threshold = 0.8
    else:
        threshold = 0.1
    if np.mean(totals) < threshold:
        import sys
        writer.add_scalar("accuracy", round(np.mean(totals) * 100, 2))
        writer.close()
        sys.exit(0)

        
    writer.add_scalar("accuracy", round(np.mean(totals) * 100, 2))
    writer.add_scalar("std", round(np.std(totals) * 100, 2))
    writer.add_text("accs", str(totals))
    
    print("accuracy:", round(np.mean(totals) * 100, 2))
    print("std:", round(np.std(totals) * 100, 2))
    print("accs:", str(totals))
    
    writer.close()
