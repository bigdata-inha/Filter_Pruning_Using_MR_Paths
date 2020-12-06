"""VGG11/13/16/19 in Pytorch."""
import os
import tqdm
import time
import numpy as np

from utils.metrics import AverageMeter, cls_accuracy
from utils.misc import timeit, print_cuda_statistics
from datasets.cifar100 import *
from modules.layers import *
from itertools import islice
from igraph import *

import torch.nn as nn
import torch.optim as optim
import torch.utils.model_zoo as model_zoo
import torch
import networkx as nx

import collections, functools, operator


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=100, batch_norm=True, init_weights=True):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name], batch_norm=batch_norm)
        self.classifier = nn.Sequential(
            nn.Linear(512 * int(32/2**5) * int(32/2**5), 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for m in cfg:
            if m == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, m, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(m), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = m
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def init_graph(self,config):
        self.best_valid_acc = 0

        self.config = config

        self.is_cuda = torch.cuda.is_available()

        self.cuda = self.is_cuda & self.config.cuda
        self.manual_seed = self.config.seed

        if self.cuda:
            self. device =torch.device("cuda")
            torch.cuda.manual_seed(self.manual_seed)
            torch.cuda.set_device(self.config.gpu_device)

            print("Program will run on *****GPU-CUDA*****\n")
            # print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.manual_seed)
            print("Program will run on *****CPU*****\n")

        self.total_linear_relevance_list = []
        self.total_conv_relevance_list = []

        self.Rp_module_list = dict()
        self.named_modules_idx_list = dict()
        self.named_modules_list = dict()
        self.named_conv_list = dict()
        self.named_conv_idx_list = dict()

        self.original_conv_output = dict()

        self.stayed_channels = dict()

        self.total_module_list = []
        self.module_list = dict()

        self.record_stayed_nodes = []

        i = 0
        for idx, m in enumerate(self.features):
            if isinstance(m, torch.nn.Conv2d):
                self.named_modules_idx_list['{}.conv'.format(i)] = idx
                self.named_modules_list['{}.conv'.format(i)] = m
                self.named_conv_idx_list['{}.conv'.format(i)] = idx
                self.named_conv_list['{}.conv'.format(i)] = m
            elif isinstance(m, torch.nn.BatchNorm2d):
                self.named_modules_idx_list['{}.bn'.format(i)] = idx
                self.named_modules_list['{}.bn'.format(i)] = m
                i += 1

        self.total_module_list = []
        self.module_list = dict()


    def load_checkpoint(self, file_path="checkpoint.pth", only_weight=False):
        """
        Latest checkpoint loader
        :param file_path: str, path of the checkpoint file
        :param only_weight: bool, load only weight or all training state
        :return:
        """
        try:
            print("Loading checkpoint '{}'".format(file_path))
            checkpoint = torch.load(file_path)
            if only_weight:
                self.load_state_dict(checkpoint)
                print("Checkpoint loaded successfully\n")
            else:
                self.current_epoch = checkpoint['epoch']
                self.current_iteration = checkpoint['iteration']
                self.load_state_dict(checkpoint['state_dict'])
                print("Checkpoint loaded successfully at (epoch {}) at (iteration {})\n"
                                 .format(checkpoint['epoch'], checkpoint['iteration']))
        except OSError as e:
            print("No checkpoint exists")

    def adjust_learning_rate(self, optimizer, epoch, iteration, num_iter):
        warmup_epoch = 5
        warmup_iter = warmup_epoch * num_iter
        current_iter = iteration + epoch * num_iter
        max_iter = 100 * num_iter

        lr = 0.1 * (1 + cos(pi * (current_iter - warmup_iter) / (max_iter - warmup_iter))) / 2

        if epoch < warmup_epoch:
            lr = 0.1 * current_iter / warmup_iter

        if iteration == 0:
            print('current learning rate:{0}'.format(lr))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def model_train(self, config):
        """
        Main training function, with per-epoch model saving
        :return:
        """
        last_conv = list(self.named_conv_list.values())[-1]
        self.classifier = torch.nn.Linear(last_conv.out_channels, 100)

        self.data_loader = Cifar100DataLoader(self.config)

        self.loss_fn = nn.CrossEntropyLoss()
        self.loss_fn = self.loss_fn.to(self.device)

        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4,
                                         nesterov=True)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.config.milestones,
                                                        gamma=self.config.gamma)

        self.to(self.device)

        history = []
        for epoch in range(self.config.max_epoch):
            self.current_epoch = epoch
            self.train_one_epoch()

            valid_acc = self.validate()
            is_best = valid_acc > self.best_valid_acc
            if is_best:
                self.best_valid_acc = valid_acc
            # self.save_checkpoint(is_best=is_best)

            history.append(valid_acc)
            self.scheduler.step(epoch)

        return self.best_valid_acc, history

    def train_one_epoch(self):
        """
        One epoch training function
        :return:
        """

        tqdm_batch = tqdm.tqdm(self.data_loader.train_loader, total=self.data_loader.train_iterations,
                               desc="Epoch-{}-".format(self.current_epoch))

        self.train()

        epoch_loss = AverageMeter()
        top1_acc = AverageMeter()
        top5_acc = AverageMeter()

        current_batch = 0
        for i, (x, y) in enumerate(tqdm_batch):
            if self.cuda:
                x, y = x.cuda(non_blocking=self.config.async_loading), y.cuda(non_blocking=self.config.async_loading)

            self.optimizer.zero_grad()
#             self.adjust_learning_rate(self.optimizer, self.current_epoch, i, self.data_loader.train_iterations)

            pred = self(x)
            cur_loss = self.loss_fn(pred, y)

            if np.isnan(float(cur_loss.item())):
                raise ValueError('Loss is nan during training...')

            cur_loss.backward()
            self.optimizer.step()

            top1, top5 = cls_accuracy(pred.data, y.data, topk=(1, 5))
            top1_acc.update(top1.item(), x.size(0))
            top5_acc.update(top5.item(), x.size(0))

            epoch_loss.update(cur_loss.item())

            self.current_iteration += 1
            current_batch += 1

        tqdm_batch.close()

        print("Training at epoch-" + str(self.current_epoch) + " | " + "loss: " + str(epoch_loss.val) +
              "\tTop1 Acc: " + str(top1_acc.val))

    def validate(self):
        """
        One epoch validation
        :return:
        """

        tqdm_batch = tqdm.tqdm(self.data_loader.valid_loader, total=self.data_loader.valid_iterations,
                               desc="Valiation at -{}-".format(self.current_epoch))

        self.eval()

        epoch_loss = AverageMeter()
        top1_acc = AverageMeter()
        top5_acc = AverageMeter()

        for x, y in tqdm_batch:
            if self.cuda:
                x, y = x.cuda(non_blocking=self.config.async_loading), y.cuda(non_blocking=self.config.async_loading)

            # model
            pred = self(x)
            # loss
            cur_loss = self.loss_fn(pred, y)
            if np.isnan(float(cur_loss.item())):
                raise ValueError('Loss is nan during validation...')

            top1, top5 = cls_accuracy(pred.data, y.data, topk=(1, 5))
            top1_acc.update(top1.item(), x.size(0))
            top5_acc.update(top5.item(), x.size(0))

            epoch_loss.update(cur_loss.item())

        print("Validation results at epoch-" + str(self.current_epoch) + " | " + "loss: " +
              str(epoch_loss.avg) + "\tTop1 Acc: " + str(top1_acc.val))

        tqdm_batch.close()

        return top1_acc.avg

    def _validate(self, config):
        """
        One epoch validation
        :return:
        """
        self.data_loader = Cifar100DataLoader(self.config)
        self.loss_fn = nn.CrossEntropyLoss()
        self.loss_fn = self.loss_fn.to(self.device)
        tqdm_batch = tqdm.tqdm(self.data_loader.valid_loader, total=self.data_loader.valid_iterations,
                               desc="Valiation at -{}-".format(self.current_epoch))

        self.eval()

        epoch_loss = AverageMeter()
        top1_acc = AverageMeter()
        top5_acc = AverageMeter()

        for x, y in tqdm_batch:
            if self.cuda:
                x, y = x.cuda(non_blocking=self.config.async_loading), y.cuda(non_blocking=self.config.async_loading)

            # model
            pred = self(x)
            # loss
            cur_loss = self.loss_fn(pred, y)
            if np.isnan(float(cur_loss.item())):
                raise ValueError('Loss is nan during validation...')

            top1, top5 = cls_accuracy(pred.data, y.data, topk=(1, 5))
            top1_acc.update(top1.item(), x.size(0))
            top5_acc.update(top5.item(), x.size(0))

            epoch_loss.update(cur_loss.item())

        print("Validation results at epoch-" + str(self.current_epoch) + " | " + "loss: " +
              str(epoch_loss.avg) + "\tTop1 Acc: " + str(top1_acc.val))

        tqdm_batch.close()

        return top1_acc.avg
def new_vgg16(num_classes, batch_norm=True):
    return VGG('VGG16', num_classes, batch_norm=batch_norm, init_weights=True)
