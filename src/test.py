import torch
import random

from pipeline import config
from pipeline.core.train_core import test_accuracy

import datasets.mnist.mnist_loader as mnist_loader
import datasets.emnist.emnist_loader as emnist_loader
import datasets.chars75k.chars_loader as chars_loader

from pipeline.core.utils import *

import args

def main(args):
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)

    device = get_device()
    if args.dataset == "MNIST":
        # --dataset MNIST --model_path ./save_model/best_{model_name}.pth
        test_loader = mnist_loader.get_test_loader(batch_size_test=config.batch_size_test)
    elif args.dataset == "CHARS":
        test_loader = chars_loader.get_test_loader(".\datasets\chars75k\chars", batch_size_test=config.batch_size_test)
    elif args.dataset == "EMNIST":
        # --dataset EMNIST --model_path ./save_model/best_{model_name}.pth
        test_loader = emnist_loader.get_test_loader(batch_size_test=config.batch_size_test)
    else:
        print("Incorrect argument")
        exit(1)

    model_name=args.model_name

    network = torch.load('./save_model_{}/best_{}.pth'.format(args.dataset, model_name)).to(device)

    acc, y_pred = test_accuracy(test_loader, network, device)
    print("acc {0:.2f}%".format(acc))

    pytorch_total_params_res = sum(p.numel() for p in network.parameters())
    print("Num of params {0}".format(pytorch_total_params_res))

if __name__ == '__main__':
    main(args.get_setup_args_test())
