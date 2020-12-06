

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


class fashion_mnist_dataloader:
    def __init__(self, BATCH_SIZE):
        # BATCH_SIZE = 64
        train_set =datasets.MNIST('./.data',
                           train=True,
                           download=True,
                           transform=transforms.Compose([
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ]))
        valid_set = datasets.MNIST('./.data',
                           train=False,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ]))


        self.train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                                       num_workers=True, pin_memory=True)
        self.valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False,
                                       num_workers=True, pin_memory=True)
        self.train_iterations = len(self.train_loader)
        self.valid_iterations = len(self.valid_loader)


class selected_fashion_mnist_dataloader:
    def __init__(self):
        train_set =datasets.MNIST('./.data',
                           train=True,
                           download=True,
                           transform=transforms.Compose([
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ]))
        valid_set = datasets.MNIST('./.data',
                           train=False,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ]))


        train_set_targets = np.array(train_set.targets)
        valid_set_targets = np.array(valid_set.targets)

        random_select_target = []

        # select validation data 1
        for i in range(10):
            rand_target_index = np.random.choice(np.where(valid_set_targets == i)[0], 1)[0]
            random_select_target.append(rand_target_index)

        # for cls in subset_labels:
        #     rand_target_index = np.random.choice(np.where(valid_set_targets == cls)[0], 1)[0]
        #     random_select_target.append(rand_target_index)

        selected_valid_set = torch.utils.data.dataset.Subset(valid_set, random_select_target)
        self.valid_loader = DataLoader(selected_valid_set, batch_size=1, shuffle=True)
        self.valid_iterations = len(self.valid_loader)