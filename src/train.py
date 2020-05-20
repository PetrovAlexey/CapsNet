from pipeline import config
from pipeline.core.train_core import good_train
#from models.resnet import resnet18
from pipeline.core.utils import *

import os
import time
import random

import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

import torchvision.models as models
from collections import OrderedDict

import datasets.mnist.mnist_loader as mnist_loader
import datasets.chars75k.chars_loader as chars_loader
import datasets.emnist.emnist_loader as emnist_loader
import datasets.etl8.etl8_loader as etl8_loader

import args

def main(args):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = get_device()

    n_classes = 10

    if args.dataset == "MNIST":
        # --dataset MNIST --model_path ./save_model/best_{model_name}.pth
        train_loader, test_loader = mnist_loader.get_train_valid_loader(config.batch_size_train, config.batch_size_test)
        n_classes = 10
    elif args.dataset == "ETL8":
        # --dataset MNIST --model_path ./save_model/best_{model_name}.pth
        train_loader, test_loader, n_classes = etl8_loader.get_train_valid_loader(".\datasets\etl8\ETL8B2C1", config.batch_size_train, config.batch_size_test)
        print (n_classes)
        #n_classes = 10
    elif args.dataset == "CHARS":
        train_loader, test_loader = chars_loader.get_train_valid_loader(".\datasets\chars75k\chars_fnt",
                                                                        config.batch_size_train,
                                                                        config.batch_size_test)
        n_classes = 63
    elif args.dataset == "EMNIST":
        train_loader, test_loader = emnist_loader.get_train_valid_loader(config.batch_size_train, config.batch_size_test)
        n_classes = 47
    else:
        print("Incorrect argument")
        exit(1)

    if args.model_name == 'resnet18':
        #if (args.load):
            #network = torch.load('./save_model/best_{}.pth'.format(args.model_name)).to(device)
        #else:
            model_name = args.model_name
            network = models.resnet18()
            fc = nn.Sequential(OrderedDict([
                ('fc1', nn.Linear(512,100)),
                ('relu', nn.ReLU()),
                ('fc2', nn.Linear(100,n_classes)),
                ('output', nn.LogSoftmax(dim=1))
            ]))
            network.fc = fc
            network = network.to(device)
    else:
        print("Incorrect argument")
        exit(1)

    model_name=args.model_name

    print('{}: Training {} (logging to /logs/{})'.format(
        1, model_name, model_name)
    )

    # optimizer = optim.Adam(network.parameters(), lr=config.init_lr, weight_decay=0)
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.001)

    optimizer = optim.Adam(network.parameters())
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    #network = torch.load('./save_model/best_{}.pth'.format(model_name)).to(device)

    if not os.path.exists('./save_model_{}/'.format(args.dataset)):
        os.mkdir('./save_model_{}/'.format(args.dataset))

    good_train(device=device, network=network,
                model_name = model_name,
                dataset = args.dataset,
                n_epochs=10,
                train_loader=train_loader, test_loader = test_loader, val_loader = test_loader,
                optimizer=optimizer, train_losses=[],
                train_counter = [])


if __name__ == '__main__':
    main(args.get_setup_args_train())