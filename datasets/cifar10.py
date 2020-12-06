"""
Cifar100 Dataloader implementation
"""
import logging
import numpy as np

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.utils.data import DataLoader


class Cifar10DataLoader:
    def __init__(self, config, subset_labels=None):
        self.config = config
        self.logger = logging.getLogger("Cifar10DataLoader")
        self.mean, self.std =[0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]

        if config.data_mode == "download":
            self.logger.info("Loading DATA.....")

            train_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)])

            valid_transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)])

            train_set = datasets.CIFAR10("./data", train=True, download=True, transform=train_transform)
            valid_set = datasets.CIFAR10("./data", train=False, transform=valid_transform)

        self.train_loader = DataLoader(train_set, batch_size=self.config.batch_size, shuffle=True)
        self.valid_loader = DataLoader(valid_set, batch_size=self.config.batch_size, shuffle=False)

        self.train_iterations = len(self.train_loader)
        self.valid_iterations = len(self.valid_loader)

    def finalize(self):
        pass
    
    
class Selected_Cifar10DataLoader: # compute RAP
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("Cifar100DataLoader")
        self.mean, self.std =[0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]

        if config.data_mode == "download":
            self.logger.info("Loading DATA.....")

            train_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)])

            valid_transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)])

            train_set = datasets.CIFAR10("./data", train=True, download=True, transform=train_transform)
            valid_set = datasets.CIFAR10("./data", train=False, transform=valid_transform)

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
            self.valid_loader = DataLoader(selected_valid_set, batch_size=config.batch_size, shuffle=True)
            self.valid_iterations= len(self.valid_loader)

        else:
            raise Exception("Please specify in the json a specified mode in data_mode")
