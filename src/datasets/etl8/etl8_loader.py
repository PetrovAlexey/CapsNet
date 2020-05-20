import bitstring
import numpy as np
from PIL import Image, ImageEnhance
from PIL import ImageOps, ImageMath
from matplotlib import pyplot as plt
import cv2

import struct
from PIL import Image

import torch
import torchvision

from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

from torch.utils.data.sampler import SubsetRandomSampler

random_seed = 42

data_transform = transform=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])


resize_transform = torchvision.transforms.Compose([
    #torchvision.transforms.RandomRotation(0, fill=(1, )),
                                                   torchvision.transforms.Grayscale(3),
                                                   torchvision.transforms.Resize((28,28))])

resize_transform_train = torchvision.transforms.Compose([
    #torchvision.transforms.RandomRotation(0, fill=(1, )),
    torchvision.transforms.Grayscale(3),
    torchvision.transforms.Resize((28,28))])

def read_record_ETL8B2(f):
    s = f.read(512)

    r = struct.unpack('>2H4s504s', bytes(s, encoding='latin1'))
    i1 = Image.frombytes('1', (64, 63), r[3], 'raw')
    return r + (i1,)

def read_code_ETL8B2(r):
    iI = Image.eval(r[-1], lambda x: not x)
    return iI, r[1]

def create_data(dataset_size, filename, val_indices):
    data =  np.zeros((dataset_size,3,28,28))
    label = np.chararray(dataset_size, itemsize=4)
    with open(filename, 'r', encoding='latin1') as f:
        for i in range(dataset_size):
            f.seek((i + 1) * 512)
            temper = read_record_ETL8B2(f)
            r, code = read_code_ETL8B2(temper)

            if (i in val_indices):
                temp = resize_transform(r)
            else:
                temp = resize_transform_train(r)
            #temp = resize_transform(r)
            temp=np.array(temp)
            temp = data_transform(temp)
            data[i,:,:] = temp
            label[i] = code
        return data, label

def get_train_valid_loader(filename, batch_size_train, batch_size_val):
    #filename = 'drive/My Drive/Диплом/ETL8B2C1'
    validation_split = .2
    shuffle_dataset = True

    # Creating data indices for training and validation splits:
    dataset_size = 51200
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    data, label = create_data(dataset_size, filename, val_indices)
    le = preprocessing.LabelEncoder()
    le.fit(label)

    data = torch.Tensor(data)

    #data = data.unsqueeze(1)


    label = torch.Tensor(le.transform(label))

    label = torch.Tensor(label)
    full_data = []
    for i in range(len(data)):
        full_data.append([data[i], label[i]])

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(full_data, batch_size=batch_size_train,
                                               sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(full_data, batch_size=batch_size_val,
                                              sampler=valid_sampler)

    n_classes = len(list(le.classes_))
    return train_loader, test_loader, n_classes

def get_test_loader(batch_size_test=256):
    #TODO: Implement test_loader
    return 0