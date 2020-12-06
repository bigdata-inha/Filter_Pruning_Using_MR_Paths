# !/usr/bin/env python
# coding: utf-8

import os
import tqdm
import time
import numpy as np

from utils.metrics import AverageMeter, cls_accuracy
from utils.misc import timeit, print_cuda_statistics
from datasets.cifar100 import *
from modules.layers import *


import torch.nn as nn
import torch.optim as optim
import torch.utils.model_zoo as model_zoo
import torch
import tqdm
import networkx as nx
from datasets.fashion_mnist import *

# ## 뉴럴넷으로 Fashion MNIST 학습하기
# 입력 `x` 는 `[배치크기, 색, 높이, 넓이]`로 이루어져 있습니다.
# `x.size()`를 해보면 `[64, 1, 28, 28]`이라고 표시되는 것을 보실 수 있습니다.
# Fashion MNIST에서 이미지의 크기는 28 x 28, 색은 흑백으로 1 가지 입니다.
# 그러므로 입력 x의 총 특성값 갯수는 28 x 28 x 1, 즉 784개 입니다.
# 우리가 사용할 모델은 3개의 레이어를 가진 인공신경망 입니다.

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.classifier = Sequential(
            Linear(784,256),
            ReLU(True),
            Linear(256,128),
            ReLU(True),
            Linear(128,10)
        )

    def init_graph(self):
        self.best_valid_acc = 0

        self.is_cuda = torch.cuda.is_available()

        self.cuda = self.is_cuda
        self.manual_seed = 0
        self.gpu_device = 0

        self.device = torch.device("cuda")
        torch.cuda.manual_seed(self.manual_seed)
        torch.cuda.set_device(0)

        self.total_module_list = []
        self.module_list = dict()

        self.Linear3to2 = set()
        self.Linear2to1 = set()
        self.Linear1to0 = set()

        cnt = 2
        for i in reversed(self.classifier):
            if isinstance(i, torch.nn.Linear):
                linear_name = 'Linear_' + str(cnt)
                self.module_list[linear_name] = []
                self.total_module_list.append(i)
                cnt -= 1

    def load_checkpoint(self, file_path='C:/Users/USER001/Desktop/Weight/weight/'):
        """
        Latest checkpoint loader
        :param file_path: str, path of the checkpoint file
        :param only_weight: bool, load only weight or all training state
        :return:
        """
        print("Loading checkpoint '{}'".format(file_path))
        checkpoint = torch.load(file_path + 'all.tar')  # dict 불러오기
        self.load_state_dict(checkpoint['model'])
        print("Checkpoint loaded successfully\n")


    def forward(self, x):
        x = x.view(-1, 784)
        x = self.classifier(x)
        return x

    def relprop(self, R):
        x, self.relevance_list = self.classifier.relprop(R)
        return x


    def weight_generating_Max(self, target_cls):
        self.NN_graph = []
        """"
        Linear(in_features=784, out_features=256, bias=True)
        Linear(in_features=256, out_features=128, bias=True)
        Linear(in_features=128, out_features=10, bias=True)
        """
        tmp = 3
        for module in self.total_module_list: #
            if tmp == 3:  # last linear
                t_node = 'linear' + str(tmp)
                end = ['linear' + str(tmp - 1) + '_' + str(i) for i in range(module.weight[target_cls, :].size()[0])]
                for connection in range(module.weight[target_cls, :].size()[0]):
                    self.NN_graph.append([t_node, end[connection], float(module.weight[target_cls, :][connection])])

            else:  # first/second linear
                start = ['linear' + str(tmp) + '_' + str(i) for i in range(module.weight.size()[0])]
                end = ['linear' + str(tmp - 1) + '_' + str(i) for i in range(module.weight.size()[1])]
                for i in range(len(start)):
                    for j in range(len(end)):
                        #                     print(float(module.weight[i,j)])
                        self.NN_graph.append([start[i], end[j], float(module.weight[i, j])])
            tmp -= 1


    def weight_generating_shortest(self,target_cls):
        NN_graph = []
        """"
        Linear(in_features=784, out_features=256, bias=True)
        Linear(in_features=256, out_features=128, bias=True)
        Linear(in_features=128, out_features=10, bias=True)
        """
        tmp = 3
        eps = 0.0001

        for module in self.total_module_list:  #
            if tmp == 3:  # last linear
                module_weight = torch.abs(1 / (module.weight + eps))[target_cls, :]
                # pw = torch.clamp(module_weight, min=0)
                t_node = 'linear' + str(tmp)
                end = ['linear' + str(tmp - 1) + '_' + str(i) for i in range(module_weight.size()[0])]
                for connection in range(module_weight.size()[0]):
                    NN_graph.append([t_node, end[connection], float(module_weight[connection])])

            else:  # first/second linear
                module_weight = torch.abs(1 / (module.weight+eps))
                # pw = torch.clamp(module_weight, min=0)
                start = ['linear' + str(tmp) + '_' + str(i) for i in range(module_weight.size()[0])]
                end = ['linear' + str(tmp - 1) + '_' + str(i) for i in range(module_weight.size()[1])]
                for i in range(len(start)):
                    for j in range(len(end)):
                        #                     print(float(module.weight[i,j)])
                        NN_graph.append([start[i], end[j], float(module_weight[i, j])])
            tmp -= 1

        G = nx.DiGraph()
        G.add_weighted_edges_from(NN_graph)

        for i in range(784):
            target_node = 'linear0_' + str(i)
            longest_path = nx.dijkstra_path(G, source='linear3', target=target_node, weight='weight')
            self.Linear3to2.add(longest_path[1])
            self.Linear2to1.add(longest_path[2])
            self.Linear1to0.add(longest_path[3])


    def relevance_generating_shortest(self):
        NN_graph = []
        """"
        Linear(in_features=784, out_features=256, bias=True)
        Linear(in_features=256, out_features=128, bias=True)
        Linear(in_features=128, out_features=10, bias=True)
        """
        tmp = 3
        eps = 0.0001
        for name, relevance in self.relevance_list.items():
            if tmp == 3:  # last linear
                module_weight = torch.abs(1 / (relevance + eps))[0,:] # [1,128]
                # pw = torch.clamp(module_weight, min=0)
                t_node = 'linear' + str(tmp)
                end = ['linear' + str(tmp - 1) + '_' + str(i) for i in range(len(module_weight))]
                for connection in range(len(module_weight)):
                    NN_graph.append([t_node, end[connection], float(module_weight[connection])])

            else:  # first/second linear
                module_weight = torch.abs(1 / (relevance + eps))
                # pw = torch.clamp(module_weight, min=0)
                start = ['linear' + str(tmp) + '_' + str(i) for i in range(module_weight.size()[0])]
                end = ['linear' + str(tmp - 1) + '_' + str(i) for i in range(module_weight.size()[1])]
                for i in range(len(start)):
                    for j in range(len(end)):
                        #                     print(float(module.weight[i,j)])
                        NN_graph.append([start[i], end[j], float(module_weight[i, j])])
            tmp -= 1

        G = nx.DiGraph()
        G.add_weighted_edges_from(NN_graph)

        for i in range(784):
            target_node = 'linear0_' + str(i)
            longest_path = nx.dijkstra_path(G, source='linear3', target=target_node, weight='weight')
            self.Linear3to2.add(longest_path[1])
            self.Linear2to1.add(longest_path[2])
            self.Linear1to0.add(longest_path[3])

    def module_surgery(self):
        pass




    # ## 테스트하기

    @timeit
    def model_train(self):
        """
        Main training function, with per-epoch model saving
        :return:
        """
        self.data_loader = fashion_mnist_dataloader(BATCH_SIZE=128)

        self.loss_fn = nn.CrossEntropyLoss()
        self.loss_fn = self.loss_fn.to(self.device)

        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005,
                                         nesterov=True)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[10,20,30],
                                                        gamma=0.9)
        self.to(self.device)
        self.current_iteration = 0

        history = []
        for epoch in range(40):
            self.current_epoch = epoch
            self.train_one_epoch()

            valid_acc = self.validate()
            is_best = valid_acc > self.best_valid_acc
            if is_best:
                self.best_valid_acc = valid_acc
            # self.save_checkpoint(is_best=is_best)

            history.append(valid_acc)
            self.scheduler.step(valid_acc)

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
        for x, y in tqdm_batch:
            if self.cuda:
                x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)

            self.optimizer.zero_grad()

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
        self.data_loader = fashion_mnist_dataloader(BATCH_SIZE=128)

        tqdm_batch = tqdm.tqdm(self.data_loader.valid_loader, total=self.data_loader.valid_iterations,
                               desc="Valiation at -{}-".format(self.current_epoch))

        self.eval()

        epoch_loss = AverageMeter()
        top1_acc = AverageMeter()
        top5_acc = AverageMeter()

        for x, y in tqdm_batch:
            if self.cuda:
                x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)

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

    def _validate(self):
        """
        One epoch validation
        :return:
        """
        current_epoch = 30 # 30 epoch학습 시킴
        self.data_loader = fashion_mnist_dataloader(BATCH_SIZE=128)
        self.loss_fn = nn.CrossEntropyLoss()
        self.loss_fn = self.loss_fn.to(self.device)
        tqdm_batch = tqdm.tqdm(self.data_loader.valid_loader, total=self.data_loader.valid_iterations,
                               desc="Valiation at -{}-".format(current_epoch))

        self.eval()

        epoch_loss = AverageMeter()
        top1_acc = AverageMeter()
        top5_acc = AverageMeter()

        for x, y in tqdm_batch:
            if self.cuda:
                x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)

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

        print("Validation results at epoch-" + str(current_epoch) + " | " + "loss: " +
              str(epoch_loss.avg) + "\tTop1 Acc: " + str(top1_acc.val))

        tqdm_batch.close()

        return top1_acc.avg




