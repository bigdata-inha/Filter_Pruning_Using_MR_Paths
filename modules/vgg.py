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

__all__ = ['vgg16_bn']

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    #    model = VGG(input_shape, num_classes, data_set)

    def __init__(self, input_shape=224, num_classes=1000, data_set='ImageNet', init_weights=True):
        super(VGG, self).__init__()
        self.data_set = data_set
        self.features = self.make_layers(cfg['D'], batch_norm=True)

        if data_set == 'ImageNet':
            self.avgpool = AdaptiveAvgPool2d((7, 7))

        self.classifier = Sequential(
            Linear(512 * int(input_shape / 2 ** 5) * int(input_shape / 2 ** 5), 4096),
            ReLU(True),
            Dropout(),
            Linear(4096, 4096),
            ReLU(True),
            Dropout(),
            Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def init_graph(self, config):
        self.best_valid_acc = 0

        self.config = config

        self.is_cuda = torch.cuda.is_available()

        self.cuda = self.is_cuda & self.config.cuda
        self.manual_seed = self.config.seed

        if self.cuda:
            self.device = torch.device("cuda")
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
        cnt = 15

        for m in reversed(self.classifier):
            if isinstance(m, torch.nn.Linear):
                linear_name = 'L_' + str(cnt)
                self.module_list[linear_name] = []
                self.total_module_list.append(m)
                cnt -= 1

        for m in reversed(self.features):
            if isinstance(m, torch.nn.Conv2d):
                conv_name = 'C_' + str(cnt)
                self.module_list[conv_name] = []
                self.total_module_list.append(m)
                cnt -= 1

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

    def forward(self, x):
        x = self.features(x)
        if self.data_set == 'ImageNet':
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def relprop(self, R):
        x, self.linear_relevance_list = self.classifier.linear_relprop(R)
        x = x.reshape_as(next(reversed(self.features._modules.values())).Y)
        # x = self.avgpool.relprop(x, alpha)
        if self.data_set == 'ImageNet':
            x = self.avgpool.RAP_relprop(x)

        x, self.conv_relevance_list = self.features.conv_relprop(x)
        # print('relevance_15 ',self.linear_relevance_list['relevance_15'].sum())
        # print('relevance_14 ',self.linear_relevance_list['relevance_14'].sum())
        # print('relevance_13 ',self.linear_relevance_list['relevance_13'].sum())
        # print('relevance_12 ',self.conv_relevance_list['relevance_12'].sum())
        # print('relevance_11 ',self.conv_relevance_list['relevance_11'].sum())
        # print('relevance_0 ',self.conv_relevance_list['relevance_0'].sum())

        return x

    def RAP_relprop(self, R):
        x1, Rp_module_list = self.classifier.classifier_RAP_relprop(R)
        if torch.is_tensor(x1) == False:
            for i in range(len(x1)):
                print(x1[i].reshape_as(next(reversed(self.features._modules.values())).Y))
                x1[i] = x1[i].reshape_as(next(reversed(self.features._modules.values())).Y)
        else:
            x1 = x1.reshape_as(next(reversed(self.features._modules.values())).Y)

        if self.data_set == 'ImageNet':
            x1 = self.avgpool.RAP_relprop(x1)
        x1, Rp_module_list = self.features.RAP_relprop(x1, Rp_module_list)

        return x1, Rp_module_list

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

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, BatchNorm2d(v), ReLU(inplace=True)]
                else:
                    layers += [conv2d, ReLU(inplace=True)]
                in_channels = v
        return Sequential(*layers)

    def k_shortest_paths(self, G, source, target, k):
        return list(islice(nx.shortest_simple_paths(G, source, target, weight='weight'), k))

    def weight_generating_shortest(self):

        for target_cls in range(100):
            NN_graph = []
            """"
            Linear(in_features=4096, out_features=100, bias=True),
            Linear(in_features=4096, out_features=4096, bias=True),
            Linear(in_features=512, out_features=4096, bias=True),
            """
            tmp = 16
            eps = 0.0001

            for module in self.total_module_list:  #
                if tmp == 16:  # last linear
                    module_weight = torch.abs(1 / (module.weight + eps))[target_cls, :].detach().cpu()
                    # pw = torch.clamp(module_weight, min=0)
                    t_node = 'l' + str(tmp)
                    end = ['l' + str(tmp - 1) + '_' + str(i) for i in range(module_weight.size()[0])]
                    for connection in range(module_weight.size()[0]):
                        NN_graph.append([t_node, end[connection], float(module_weight[connection])])

                elif tmp == 15:
                    module_weight = torch.abs(1 / (module.weight + eps)).detach().cpu()
                    # pw = torch.clamp(module_weight, min=0)
                    start = ['l' + str(tmp) + '_' + str(i) for i in range(module_weight.size()[0])]
                    end = ['l' + str(tmp - 1) + '_' + str(i) for i in range(module_weight.size()[1])]
                    for i in range(len(start)):
                        for j in range(len(end)):
                            #                     print(float(module.weight[i,j)])
                            NN_graph.append([start[i], end[j], np.float(module_weight[i, j])])

                elif tmp == 14:  # first/second linear
                    module_weight = torch.abs(1 / (module.weight + eps)).detach().cpu()
                    # pw = torch.clamp(module_weight, min=0)
                    start = ['l' + str(tmp) + '_' + str(i) for i in range(module_weight.size()[0])]
                    end = ['c' + str(tmp - 1) + '_' + str(i) for i in range(module_weight.size()[1])]
                    for i in range(len(start)):
                        for j in range(len(end)):
                            #                     print(float(module.weight[i,j)])
                            NN_graph.append([start[i], end[j], np.float(module_weight[i, j])])

                else:
                    module_weight = torch.abs(1 / (module.weight + eps)).sum((2, 3)).detach().cpu()
                    start = ['c' + str(tmp) + '_' + str(i) for i in range(module_weight.size()[0])]
                    end = ['c' + str(tmp - 1) + '_' + str(i) for i in range(module_weight.size()[1])]
                    for i in range(len(start)):
                        for j in range(len(end)):
                            #                     print(float(module.weight[i,j)])
                            NN_graph.append([start[i], end[j], np.float(module_weight[i, j])])
                tmp -= 1

            G = nx.DiGraph()
            G.add_weighted_edges_from(NN_graph)

            # top - 1 shortest path
            # for i in range(3): #
            #     target_node = 'c0_' + str(i)
            #     shotest_path = nx.dijkstra_path(G, source='l16', target=target_node, weight='weight')
            #     self.record_stayed_nodes.append(shotest_path)

            # top - k shortest path
            for i in range(3):
                target_node = 'c0_' + str(i)
                self.record_stayed_nodes.append(self.k_shortest_paths(G, 'l16', target_node, k=64))

    def relevance_generating_shortest(self):

        NN_graph = []
        """"
        Linear(in_features=784, out_features=256, bias=True)
        Linear(in_features=256, out_features=128, bias=True)
        Linear(in_features=128, out_features=10, bias=True)
        """
        tmp = 16
        eps = 0.0001
        for name, relevance in self.linear_relevance_list.items():
            if tmp == 16:  # last linear
                module_relevance_weight = torch.abs(1 / (relevance + eps))[0, :]  # (1,4096) --> [4096]
                # pw = torch.clamp(module_weight, min=0)
                t_node = 'l' + str(tmp)
                end = ['l' + str(tmp - 1) + '_' + str(i) for i in range(len(module_relevance_weight))]
                for connection in range(len(module_relevance_weight)):
                    NN_graph.append([t_node, end[connection], float(module_relevance_weight[connection])])

            elif tmp == 15:  # first/second linear
                module_relevance_weight = torch.abs(1 / (relevance + eps))
                # pw = torch.clamp(module_weight, min=0)
                start = ['l' + str(tmp) + '_' + str(i) for i in range(module_relevance_weight.size()[0])]
                end = ['l' + str(tmp - 1) + '_' + str(i) for i in range(module_relevance_weight.size()[1])]
                for i in range(len(start)):
                    for j in range(len(end)):
                        #                     print(float(module.weight[i,j)])
                        NN_graph.append([start[i], end[j], np.float(module_relevance_weight[i, j])])
            else:
                module_relevance_weight = torch.abs(1 / (relevance + eps))
                # pw = torch.clamp(module_weight, min=0)
                start = ['l' + str(tmp) + '_' + str(i) for i in range(module_relevance_weight.size()[0])]
                end = ['c' + str(tmp - 1) + '_' + str(i) for i in range(module_relevance_weight.size()[1])]
                for i in range(len(start)):
                    for j in range(len(end)):
                        #                     print(float(module.weight[i,j)])
                        NN_graph.append([start[i], end[j], np.float(module_relevance_weight[i, j])])
            tmp -= 1

        for name, relevance in self.conv_relevance_list.items():
            module_relevance_weight = torch.abs(1 / (relevance + eps))
            start = ['c' + str(tmp) + '_' + str(i) for i in range(module_relevance_weight.size()[0])]
            end = ['c' + str(tmp - 1) + '_' + str(i) for i in range(module_relevance_weight.size()[1])]
            for i in range(len(start)):
                for j in range(len(end)):
                    #                     print(float(module.weight[i,j)])
                    NN_graph.append([start[i], end[j], np.float(module_relevance_weight[i, j])])
            tmp -= 1

        G = nx.DiGraph()
        G.add_weighted_edges_from(NN_graph)

        # top - 1 shortest path
        # for i in range(3): #
        #     target_node = 'c0_' + str(i)
        #     shotest_path = nx.dijkstra_path(G, source='l16', target=target_node, weight='weight')
        #     self.record_stayed_nodes.append(shotest_path)

        # top - k shortest path
        for i in range(3):
            target_node = 'c0_' + str(i)
            self.record_stayed_nodes.append(self.k_shortest_paths(G, 'l16', target_node, k=64))

    def record_relevance(self):
        # print('relevance_15 ',self.linear_relevance_list['relevance_15'].sum())
        self.total_linear_relevance_list.append(self.linear_relevance_list)
        self.total_conv_relevance_list.append(self.conv_relevance_list)

    def relevance_generating_shortest_igraph(self):

        l_counter = collections.Counter()
        for d in self.total_linear_relevance_list:
            l_counter.update(d)

        c_counter = collections.Counter()
        for d in self.total_conv_relevance_list:
            c_counter.update(d)

        linear_relevance_list = dict(l_counter)
        conv_relevance_list = dict(c_counter)

        NN_graph = []
        """"
        Linear(in_features=784, out_features=256, bias=True)
        Linear(in_features=256, out_features=128, bias=True)
        Linear(in_features=128, out_features=10, bias=True)
        """
        tmp = 16
        eps = 0.0001
        for name, _relevance in linear_relevance_list.items():
            relevance = _relevance / self.config.images_per_class
            # print(name, relevance.sum())

            if tmp == 16:  # last linear
                module_relevance_weight = torch.abs(1 / (relevance + eps))[0, :]  # (1,4096) --> [4096]
                # pw = torch.clamp(module_weight, min=0)
                t_node = 'l' + str(tmp)
                end = ['l' + str(tmp - 1) + '_' + str(i) for i in range(len(module_relevance_weight))]
                for connection in range(len(module_relevance_weight)):
                    NN_graph.append((t_node, end[connection], float(module_relevance_weight[connection])))

            elif tmp == 15:  # first/second linear
                module_relevance_weight = torch.abs(1 / (relevance + eps))
                # pw = torch.clamp(module_weight, min=0)
                start = ['l' + str(tmp) + '_' + str(i) for i in range(module_relevance_weight.size()[0])]
                end = ['l' + str(tmp - 1) + '_' + str(i) for i in range(module_relevance_weight.size()[1])]
                for i in range(len(start)):
                    for j in range(len(end)):
                        #                     print(float(module.weight[i,j)])
                        NN_graph.append((start[i], end[j], np.float(module_relevance_weight[i, j])))
            else:
                module_relevance_weight = torch.abs(1 / (relevance + eps))
                # pw = torch.clamp(module_weight, min=0)
                start = ['l' + str(tmp) + '_' + str(i) for i in range(module_relevance_weight.size()[0])]
                end = ['c' + str(tmp - 1) + '_' + str(i) for i in range(module_relevance_weight.size()[1])]
                for i in range(len(start)):
                    for j in range(len(end)):
                        #                     print(float(module.weight[i,j)])
                        NN_graph.append((start[i], end[j], np.float(module_relevance_weight[i, j])))
            tmp -= 1

        for name, _relevance in conv_relevance_list.items():
            relevance = _relevance / self.config.images_per_class
            # print(name, relevance.sum())

            module_relevance_weight = torch.abs(1 / (relevance + eps))
            start = ['c' + str(tmp) + '_' + str(i) for i in range(module_relevance_weight.size()[0])]
            end = ['c' + str(tmp - 1) + '_' + str(i) for i in range(module_relevance_weight.size()[1])]
            for i in range(len(start)):
                for j in range(len(end)):
                    #                     print(float(module.weight[i,j)])
                    NN_graph.append((start[i], end[j], np.float(module_relevance_weight[i, j])))
            tmp -= 1

        Refined_NN_graph = [(start, end, paths) for start, end, paths in NN_graph if
                            paths < 10000]  # delete low relevance
        #         print(Refined_NN_graph)
        g = Graph.TupleList(Refined_NN_graph, directed=True, weights=True)
        #         g.es.select(weight=1000000).delete()
        for i in range(3):
            tmp_g = g.copy()
            tmp_list = []
            for j in range(20):
                target_node = 'c0_' + str(i)
                shortest_path = \
                tmp_g.get_shortest_paths(v='l16', to=target_node, weights=tmp_g.es['weight'], mode=OUT, output='vpath')[
                    0]
                res = [tmp_g.vs[n]['name'] for n in shortest_path]
                tmp_list.append(res)
                shortest_edge = \
                tmp_g.get_shortest_paths(v='l16', to=target_node, weights=tmp_g.es['weight'], mode=OUT, output='epath')[
                    0]
                tmp_g.delete_edges(shortest_edge)
            self.record_stayed_nodes.append(tmp_list)

        # reinitialization relevance_list
        self.total_linear_relevance_list = []
        self.total_conv_relevance_list = []

    # def record_conv_output(self,config):
    #     self.data_loader = Cifar100DataLoader(self.config)
    #
    #     inputs, _ = next(iter(self.data_loader.train_loader))
    #     if self.cuda:
    #         inputs = inputs.cuda(non_blocking=self.config.async_loading)
    #     x = inputs
    #     i = 0
    #     for m in self.features:
    #         x = m(x)
    #         if isinstance(m, torch.nn.Conv2d):
    #             self.original_conv_output['{}.conv'.format(i)] = x.data
    #             i += 1

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

    @timeit
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


def vgg16_bn(input_shape=224, num_classes=1000, data_set='ImageNet'):
    """VGG 16-layer model (configuration "D") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    print(input_shape, num_classes, data_set)
    model = VGG(input_shape, num_classes, data_set)
    # if pretrained:
    #     # model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']))
    #     model.load_state_dict(checkpoint['state_dict'])

    return model


