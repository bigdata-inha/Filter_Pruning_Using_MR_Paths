"""
Cifar100 Dataloader implementation
"""
import logging
import numpy as np
import copy
import random
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.utils.data import DataLoader


class Cifar100DataLoader:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("Cifar100DataLoader")
        self.mean, self.std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        random.seed(config.seed)
        np.random.seed(config.seed)

        if config.data_mode == "download":
            self.logger.info("Loading DATA.....")

            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)])

            valid_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)])

            train_set = datasets.CIFAR100("./data", train=True, download=True, transform=train_transform)
            valid_set = datasets.CIFAR100("./data", train=False, transform=valid_transform)
        else:
            raise Exception("Please specify in the json a specified mode in data_mode")

        self.train_loader = DataLoader(train_set, batch_size=self.config.batch_size, shuffle=True,
                                       num_workers=self.config.data_loader_workers, pin_memory=self.config.pin_memory)
        self.valid_loader = DataLoader(valid_set, batch_size=self.config.batch_size, shuffle=False,
                                       num_workers=self.config.data_loader_workers, pin_memory=self.config.pin_memory)
        self.train_iterations = len(self.train_loader)
        self.valid_iterations = len(self.valid_loader)

    def finalize(self):
        pass

class CURL_Cifar100DataLoader:
    def __init__(self, config):
        self.config = config
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        random.seed(config.seed)
        np.random.seed(config.seed)
        self.logger = logging.getLogger("Cifar100DataLoader")
        self.mean, self.std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]

        self.logger.info("Loading DATA.....")

        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)])

        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)])

        train_set = datasets.CIFAR100("./data", train=True, download=True, transform=train_transform)
        valid_set = datasets.CIFAR100("./data", train=False, transform=valid_transform)

        train_set_targets = np.array(train_set.targets)
        valid_set_targets = np.array(valid_set.targets)

        random_select_target = []

        for i in range(100):
            rand_target_index = np.random.choice(np.where(train_set_targets == i)[0], 256)[0]
            # rand_target_index = np.where(train_set_targets == i)[0][0]
            random_select_target.append(rand_target_index)

        selected_valid_set = torch.utils.data.dataset.Subset(train_set, random_select_target)
        self.CURL_train_loader = DataLoader(selected_valid_set, batch_size=256, shuffle=True)
        self.CURL_Ctrain_iterations = len(self.CURL_train_loader)
        self.valid_loader = DataLoader(valid_set, batch_size=128, shuffle=True, pin_memory=True)

class Selected_Cifar100DataLoader:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("Cifar100DataLoader")
        self.mean, self.std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        random.seed(config.seed)
        np.random.seed(config.seed)

        if config.data_mode == "download":
            self.logger.info("Loading DATA.....")

            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)])

            valid_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)])

            train_set = datasets.CIFAR100("./data", train=True, download=True, transform=train_transform)
            valid_set = datasets.CIFAR100("./data", train=False, transform=valid_transform)

            train_set_targets = np.array(train_set.targets)
            valid_set_targets = np.array(valid_set.targets)

            random_select_target = []

            # for cls in range(100):
            #     target_index = list(np.where(train_set_targets == cls)[0])
            #     random_select_target += target_index

            for i in range(100):
                rand_target_index = list(np.random.choice(list(np.where(train_set_targets == i))[0], 3))
                # rand_target_index = np.where(train_set_targets == i)[0][0]
                # random_select_target.append(rand_target_index)
                random_select_target += rand_target_index

            selected_valid_set = torch.utils.data.dataset.Subset(train_set, random_select_target)
            self.valid_loader = DataLoader(selected_valid_set, batch_size=config.batch_size, shuffle=False)
            self.valid_iterations= len(self.valid_loader)

        else:
            raise Exception("Please specify in the json a specified mode in data_mode")

class SpecializedCifar100DataLoader:
    def __init__(self, config, subset_labels):
        self.config = config
        self.logger = logging.getLogger("Cifar100DataLoader")
        self.mean, self.std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]

        if config.data_mode == "download":
            self.logger.info("Loading DATA.....")

            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)])

            valid_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)])

            binary_train_set = datasets.CIFAR100("./data", train=True, download=True, transform=train_transform)
            binary_valid_set = datasets.CIFAR100("./data", train=False, transform=valid_transform)

            train_set = copy.deepcopy(binary_train_set)

            mapping_table = dict()  # 0은 others labels (dustbin class), 그 후 subset label은 1부터 순서대로 mapping

            for new_label, label in enumerate(subset_labels):
                mapping_table[label] = new_label + 1

            # classes without subset
            oth_cls = np.setdiff1d(np.unique(binary_train_set.targets), np.array(subset_labels,
                                                                                 dtype=np.int32))  # subset_labels = [1,2,3] -- oth_cls = 97
            train_subset_indices = [i for i, e in enumerate(binary_train_set.targets) if
                                    e in subset_labels]  # 500 * 3 = 1500
            train_num_sample = len(train_subset_indices) // len(oth_cls)  # 1500 // 97 = 15
            d = list()
            for tr in oth_cls:  # oth_cls = 97
                d += list(np.where(binary_train_set.targets == tr)[0][:train_num_sample])  # 15개씩 뽑음

            tr_oth = d  # 1500 // 97 * 97 = 1455

            # change label to binary label
            _binary_train_set = [(x, mapping_table[y]) if y in subset_labels else (x, 0) for x, y in
                                 binary_train_set]

            binary_train_set = torch.utils.data.dataset.Subset(_binary_train_set,
                                                               train_subset_indices + list(tr_oth))
            self.binary_train_loader = DataLoader(binary_train_set, batch_size=config.batch_size, shuffle=True)

            # make valid set for binary train
            valid_subset_indices = [i for i, e in enumerate(binary_valid_set.targets) if
                                    e in subset_labels]  # subset_lables = [1,2,3] -- oth_cls = 97
            valid_num_sample = len(valid_subset_indices) // len(oth_cls)  # 300 // 97 = 3
            d = list()
            for val in oth_cls:  # oth_cls = 97
                d += list(np.where(binary_valid_set.targets == val)[0][:valid_num_sample])
            vl_oth = d  # 3 * 97 = 291
            _binary_valid_set = [(x, mapping_table[y]) if y in subset_labels else (x, 0) for x, y in
                                 binary_valid_set]
            sub_binary_valid_set = torch.utils.data.dataset.Subset(_binary_valid_set,
                                                                   valid_subset_indices + list(vl_oth))
            self.binary_valid_loader = DataLoader(sub_binary_valid_set, batch_size=config.batch_size)

            # set part train set
            _part_train_set = [(x, mapping_table[y]) if y in subset_labels else (x, 0) for x, y in train_set]
            part_train_set = torch.utils.data.dataset.Subset(_part_train_set, train_subset_indices)
            self.part_train_loader = DataLoader(part_train_set, batch_size=config.batch_size, shuffle=False)

            self.binary_train_iterations = len(self.binary_train_loader)
            self.binary_valid_iterations = len(self.binary_valid_loader)
            self.part_train_iterations = len(self.part_train_loader)

        else:
            raise Exception("Please specify in the json a specified mode in data_mode")

class CURL_Cifar100DataLoader:
    def __init__(self):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        random.seed(config.seed)
        np.random.seed(config.seed)
        self.logger = logging.getLogger("Cifar100DataLoader")
        self.mean, self.std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]

        self.logger.info("Loading DATA.....")

        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)])

        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)])

        train_set = datasets.CIFAR100("./data", train=True, download=True, transform=train_transform)
        valid_set = datasets.CIFAR100("./data", train=False, transform=valid_transform)

        train_set_targets = np.array(train_set.targets)
        valid_set_targets = np.array(valid_set.targets)

        random_select_target = []

        for i in range(100):
            rand_target_index = np.random.choice(np.where(train_set_targets == i)[0], 256)[0]
            # rand_target_index = np.where(train_set_targets == i)[0][0]
            random_select_target.append(rand_target_index)

        selected_valid_set = torch.utils.data.dataset.Subset(train_set, random_select_target)
        self.CURL_train_loader = DataLoader(selected_valid_set, batch_size=256, shuffle=True)
        self.CURL_Ctrain_iterations = len(self.CURL_train_loader)
        self.valid_loader = DataLoader(valid_set, batch_size=128, shuffle=True, pin_memory=True)
