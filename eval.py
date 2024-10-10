from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.utils.data as data
import torchvision.transforms as transforms
import os
from networks import methods
from data import datasets
import numpy as np
import pandas as pd
import time


parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str, help='Optional ID dataset: cifar10 | cifar100 | imagenet', default='cifar10')
parser.add_argument('-model_arch', type=str, help='Optional model: resnet | wresnet | resnetv2', default='resnet')
parser.add_argument('-model_name', type=str, help='Optional model: resnet34 | wrn_40_2 | BiT-S-R101x1', default='resnet34')
parser.add_argument('-cal_method', type=str, 
                    help='Optional method: cal_grad_value | cal_zero', 
                    default='cal_grad_value')
parser.add_argument('-hook', type=str, help='hook type', default='bn')
parser.add_argument('-score', type=str, help='score method', default='GAIA')
parser.add_argument('-data_dir', type=str, help='Data load path', default='./data')
parser.add_argument('-model_path', type=str, help='Model load path', default='./checkpoint/models/cifar10_resnet34.pth')
parser.add_argument('-save_dir', type=str, help='Data save path', default='./records')

parser.add_argument('-batch_size', type=int, help='Batch size', default=64)
parser.add_argument('-num_workers', type=int, help='Num_workers', default=4)
parser.add_argument('-cuda', type=int, help='cuda use', default='0')
parser.add_argument('-num_classes', type=int, help='number of classes', default=10)

parser.add_argument('--loss_method', type=str, help='loss function', default='CE', choices=['CE', 'MAX'])
parser.add_argument('--alpha', type=str, help='alpha', default="1/255")
parser.add_argument('--method', type=str, help='method', default='multiply', choices=['multiply', 'add'])
parser.add_argument('--type', type=str, help='type', default='S-I', choices=['S-I', 'baseline'])

parser.add_argument("--idx", type=int, default=0)


args = parser.parse_args()
torch.cuda.set_device(args.cuda)


print(args)

if "/" in args.alpha:
    args.alpha = args.alpha.split("/")
    args.alpha = float(args.alpha[0]) / float(args.alpha[1])
else:
    args.alpha = float(args.alpha)
args.save_dir = args.save_dir if args.type == 'S-I' else args.save_dir + '_baseline'

args.save_dir = os.path.join(args.save_dir, f"{args.dataset}_{args.model_name}_{args.loss_method}_{args.alpha}_{args.method}")
os.makedirs(args.save_dir, exist_ok=True)
evaluator = methods.Methods(args)
in_test, ood_datasets, ood_name = datasets.get_datasets(args)
evaluator.get_scores(in_test, ood_name, ood_datasets)