from __future__ import division, print_function

import sys

import random
import math

import numpy as np
import os
from ash import ash_p
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from torch.autograd import Variable, grad
import torchvision
from torchvision import datasets, transforms
from torch.nn.parameter import Parameter
from collections import OrderedDict

import warnings

warnings.filterwarnings('ignore')



def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        t = self.conv1(x)
        # out = self.bn1(t)
        out = F.relu(self.bn1(t))
        # t = self.conv2(out)
        out = self.bn2(self.conv2(out))
        t = self.shortcut(x)
        out += t
        out = F.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        # self.conv1 = conv3x3(3,64)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = F.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        self.feat2 = out
        out = self.layer4(out)
        self.feat1 = out
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        self.before_head_data = out
        y = self.linear(out)
        return y
    
    def intermediate_forward(self, x, layer_index=None):
        if layer_index == 'all':
            out_list = []
            out = self.bn1(self.conv1(x))
            out = F.relu(out)
            out_list.append(out)
            out = self.layer1(out)
            out_list.append(out)
            out = self.layer2(out)
            out_list.append(out)
            out = self.layer3(out)
            out_list.append(out)
            out = self.layer4(out)
            out_list.append(out)
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out_list.append(out)
            y = self.linear(out)
            return y, out_list

        out = self.bn1(self.conv1(x))
        out = F.relu(out)
        if layer_index == 1:
            out = self.layer1(out)
        elif layer_index == 2:
            out = self.layer1(out)
            out = self.layer2(out)
        elif layer_index == 3:
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
        elif layer_index == 4:
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
        return out

    def load_state(self, path):
        tm = torch.load(path) 
        print(type(tm))
        self.load_state_dict(tm['state_dict'])
        # self.load_state_dict(tm)

    def save(self, model,
             path="../../../models/mgi/",
             filename="cifar10_resnet34_weight.pth"):
        if not os.path.exists(path):
            os.makedirs(path)
        filename = path + filename
        torch.save(model, filename)

    def save_image(self, pic, path):
        torchvision.utils.save_image(pic, path)
        

class ResNet_ASH(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_ASH, self).__init__()
        self.in_planes = 64

        # self.conv1 = conv3x3(3,64)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = F.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        self.feat2 = out
        out = self.layer4(out)
        self.feat1 = out
        out = F.avg_pool2d(out, 4)
        out = ash_p(out,percentile=70)
        out = out.view(out.size(0), -1)
        self.before_head_data = out
        y = self.linear(out)
        return y

    def load_state(self, path):
        tm = torch.load(path) 
        print(type(tm))
        self.load_state_dict(tm['state_dict'])
        # self.load_state_dict(tm)

    def save(self, model,
             path="../../../models/mgi/",
             filename="cifar10_resnet34_weight.pth"):
        if not os.path.exists(path):
            os.makedirs(path)
        filename = path + filename
        torch.save(model, filename)

    def save_image(self, pic, path):
        torchvision.utils.save_image(pic, path)

def resnet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def resnet34(num_classes=10):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)

def resnet50(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)

KNOWN_MODELS = OrderedDict([
    ('resnet18', lambda *a, **kw: ResNet(BasicBlock, [2, 2, 2, 2], *a, **kw)),
    ('resnet34', lambda *a, **kw: ResNet(BasicBlock, [3, 4, 6, 3], *a, **kw)),
    ('resnet50', lambda *a, **kw: ResNet(Bottleneck, [3, 4, 6, 3], *a, **kw)),
    ('resnet18_ash', lambda *a, **kw: ResNet_ASH(BasicBlock, [2, 2, 2, 2], *a, **kw)),
    ('resnet34_ash', lambda *a, **kw: ResNet_ASH(BasicBlock, [3, 4, 6, 3], *a, **kw)),
    ('resnet50_ash', lambda *a, **kw: ResNet_ASH(Bottleneck, [3, 4, 6, 3], *a, **kw)),
])