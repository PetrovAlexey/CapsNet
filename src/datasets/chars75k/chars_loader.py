import torch
import torchvision
import numpy as np
from .make_hdf5 import *

def get_train_valid_loader(filename, batch_size_train, batch_size_val):
    full_data, full_label = read_many_hdf5(filename)
    dataset = []
    for i in range(len(full_data)):
            dataset.append([torch.tensor(np.concatenate((full_data[i], full_data[i], full_data[i]))), full_label[i]])
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size_val)

    return train_loader, test_loader

def get_test_loader(filename, batch_size_test=256):
    #TODO: make test_loader
    full_data, full_label = read_many_hdf5(filename)
    dataset = []
    for i in range(len(full_data)):
        dataset.append([torch.tensor(np.concatenate((full_data[i], full_data[i], full_data[i]))), full_label[i]])
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size_test)

    return test_loader