import torch
import numpy as np
import torch.nn as nn
import time
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Sampler
from torch.utils.tensorboard import SummaryWriter

from model import resnet34, resnet18, MLP, Logistic
from train_args import get_args, seed_torch


def load_dataset(name, location, train=True):
    # Todo: custom for name in {'MNIST', 'CIFAR10', ...}
    name = name.lower()
    if name == 'mnist':
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST(root=location, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root=location, train=False, download=True, transform=transform)

    elif name == 'cifar10':
        transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = datasets.CIFAR10(root=location, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(root=location, train=False, download=True, transform=transform_test)

    else:
        raise ValueError(name + ' is not known.')

    return train_dataset, test_dataset


def load_model(name=None, dataset=None):
    name = name.lower()
    dataset = dataset.lower()
    if name == 'resnet18' and dataset == 'cifar10':
        net = resnet18(10)
    elif name == 'resnet34' and dataset == 'cifar10':
        net = resnet34(10)
    elif name == 'mlp' and dataset == 'mnist':
        net = MLP()
    elif name == 'logistic':
        net = Logistic()
    else:
        raise ValueError(name + ' is not known.')

    return net


def inference(model, train_loader, test_loader, device, test=True):
    correct = 0
    total = 0
    loss = 0
    with torch.no_grad():
        if test:
            for data, labels in test_loader:
                data, labels = data.to(device), labels.to(device)
                outputs, batch_loss = model(data, labels)
                loss += batch_loss * labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        else:
            for data, labels in train_loader:
                data, labels = data.to(device), labels.to(device)
                outputs, batch_loss = model(data, labels)
                loss += batch_loss * labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
    total_loss = loss / total
    total_correct = correct / total

    return total_loss, total_correct


def evaluation(epoch, num_epochs, iteration, model, train_loader, test_loader, device, tb_writer):
    print("Train process")
    train_loss, train_acc = inference(model, train_loader, test_loader, device, test=False)
    print('Epoch [{}/{}], iteration {}, train loss: {:.6f}, train acc: {:.4f}'
          .format(epoch + 1, num_epochs, iteration, train_loss, train_acc))

    print("Test process")
    test_loss, test_acc = inference(model, train_loader, test_loader, device, test=True)
    print('Epoch [{}/{}], iteration {}, test loss: {:.6f}, test acc: {:.4f}'
          .format(epoch + 1, num_epochs, iteration, test_loss, test_acc))

    tb_writer.add_scalar(f'Eval/TrainLoss', train_loss, iteration)
    tb_writer.add_scalar(f'Eval/TestLoss', test_loss, iteration)
    tb_writer.add_scalar(f'Eval/TrainAcc', train_acc, iteration)
    tb_writer.add_scalar(f'Eval/TestAcc', test_acc, iteration)


def run(**kwargs):
    num_epochs = kwargs['epochs']
    batch_size = kwargs['batch_size']
    learning_rate = kwargs['lr']
    hb_beta = kwargs['beta']
    tb_writer = kwargs['tb_writer']
    weight_decay = kwargs['weight_decay']

    # shb_dw para
    alpha = kwargs['alpha']
    hb_beta_dw = kwargs['beta']
    hb_beta_min = kwargs['beta_min']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset, test_dataset = load_dataset(kwargs['dataset'], kwargs['location'])
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    model = load_model(name=kwargs['model'], dataset=kwargs['dataset'])
    model = model.to(device)
    print(model)

    parameter_hb_0 = [param.clone().detach() for param in model.parameters()]
    parameter_hb_1 = [param.clone().detach() for param in model.parameters()]

    total_step = len(train_loader)
    curr_lr = learning_rate
    iteration = 0

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            model.train()
            images = images.to(device)
            labels = labels.to(device)
            outputs, loss = model(images, labels)
            # Backward and optimize
            model.zero_grad()
            loss.backward()

            optimizer = kwargs['optimizer'].lower()
            if optimizer == 'sgd':
                optimizer_sgd = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                optimizer_sgd.step()

            elif optimizer == 'adam':
                optimizer_adam = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                optimizer_adam.step()

            elif optimizer == 'shb':
                tb_writer.add_scalar(f'HB/Beta', hb_beta, iteration)
                for param_1, param_2 in zip(parameter_hb_1, model.parameters()):
                    param_1.data = param_2.clone().detach()
                for param, param_hb in zip(model.parameters(), parameter_hb_0):
                    param.data = (1.0 - weight_decay * curr_lr) * param.data - curr_lr * param.grad.data \
                                 + hb_beta * (param.data - param_hb)
                for param_0, param_1 in zip(parameter_hb_0, parameter_hb_1):
                    param_0.data = param_1.clone().detach()

            elif optimizer == 'shb_dw':
                # shb_dw
                hb_beta_dw = max(alpha * hb_beta_dw, hb_beta_min)
                tb_writer.add_scalar(f'HB/Beta', hb_beta_dw, iteration)
                for param_1, param_2 in zip(parameter_hb_1, model.parameters()):
                    param_1.data = param_2.clone().detach()
                for param, param_hb in zip(model.parameters(), parameter_hb_0):
                    param.data = (1.0 - weight_decay * curr_lr) * param.data - curr_lr * param.grad.data \
                                 + hb_beta_dw * (param.data - param_hb)
                for param_0, param_1 in zip(parameter_hb_0, parameter_hb_1):
                    param_0.data = param_1.clone().detach()

            else:
                raise ValueError(optimizer + ' is not known.')

            if iteration % 100 == 0:
                tb_writer.add_scalar(f'Train/Train_Loss', loss, iteration)
                print("Epoch [{}/{}], Step [{}/{}], iteration: {}, Loss: {:.4f}"
                      .format(epoch + 1, num_epochs, i + 1, total_step, iteration, loss.item()))
                model.eval()
                evaluation(epoch, num_epochs, iteration, model, train_loader, test_loader, device, tb_writer)

            iteration = iteration + 1


if __name__ == '__main__':
    args = get_args()
    seed_torch(args.seed)
    task_name = f'{args.model}_{args.dataset}/runs_bs-{args.batch_size}_lr-{args.lr}_wd-{args.weight_decay}' \
                f'_epoch-{args.epochs}_seed-{args.seed}_optimizer-{args.optimizer}' \
                f'_beta-{args.beta}-min-{args.beta_min}_alpha-{args.alpha}/'
    args.tb_writer = SummaryWriter(log_dir=args.logdir + task_name)
    kwargs = vars(args)

    print('------------------------ arguments ------------------------', flush=True)
    str_list = []
    for arg in kwargs:
        dots = '.' * (48 - len(arg))
        str_list.append('  {} {} {}'.format(arg, dots, getattr(args, arg)))
    for arg in sorted(str_list, key=lambda x: x.lower()):
        print(arg, flush=True)
    print('-------------------- end of arguments ---------------------', flush=True)

    run(**kwargs)
