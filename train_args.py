import argparse
import random
import os
import numpy as np
import torch


def get_args():
    dataset_names = ['MNIST', 'CIFAR10']
    model_names = ['ResNet18', 'ResNet34', 'MLP', 'Logistic']
    parser = argparse.ArgumentParser(description="meta config of experiment")
    parser.add_argument('--model', default='Logistic', type=str, metavar='model', choices=model_names)
    parser.add_argument('--dataset', default='MNIST', type=str, metavar='data', choices=dataset_names)
    parser.add_argument('--location', default='./dataset', type=str, help='data location')
    parser.add_argument('--optimizer', default='SHB_DW', type=str, metavar='optimizer')
    parser.add_argument('--epochs', default=60, type=int, metavar='N', help='number of epochs')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--batch_size', default=256, type=int, metavar='b', help='batch size per worker')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--beta', default=0.999, type=float, help='heavy ball parameter')
    parser.add_argument('--alpha', default=0.999, type=float, help='descending warmup parameter')
    parser.add_argument('--beta_min', default=0.1, type=float, help='heavy ball parameter')
    parser.add_argument('--weight_decay', default=0.01, type=float, help='weight_decay')
    parser.add_argument('--logdir', default='./runs/', type=str)
    args = parser.parse_args()

    return args


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
