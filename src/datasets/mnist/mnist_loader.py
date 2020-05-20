import torch
import torchvision
from sklearn.model_selection import train_test_split

def get_train_valid_loader(batch_size_train, batch_size_val):
    mnist_trainset = torchvision.datasets.MNIST(root='./files/', train=True, download=True,
                                                transform=torchvision.transforms.Compose([
                                                    torchvision.transforms.Grayscale(3),
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(
                                                        (0.1307,), (0.3081,)),
                                                ]))

    train_size = int(0.9 * len(mnist_trainset))
    test_size = len(mnist_trainset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(mnist_trainset, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size_train, shuffle=True)

    val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size_val, shuffle=True)

    # train_loader = torch.utils.data.DataLoader(
    #     torchvision.datasets.MNIST('/files/', train=True, download=True,
    #                                transform=torchvision.transforms.Compose([
    #                                    torchvision.transforms.Grayscale(3),
    #                                    torchvision.transforms.ToTensor(),
    #                                    torchvision.transforms.Normalize(
    #                                        (0.1307,), (0.3081,)),
    #                                ])
    #                                ),
    #     batch_size=batch_size_train, shuffle=True)
    #
    # val_loader = torch.utils.data.DataLoader(
    #     torchvision.datasets.MNIST('/files/', train=False, download=True,
    #                                transform=torchvision.transforms.Compose([
    #                                    torchvision.transforms.Grayscale(3),
    #                                    torchvision.transforms.RandomRotation(15, fill=0),
    #                                    torchvision.transforms.ToTensor(),
    #                                    torchvision.transforms.Normalize(
    #                                        (0.1307,), (0.3081,))
    #                                ])
    #                                ),
    #     batch_size=batch_size_val, shuffle=True)
    return train_loader, val_loader

def get_test_loader(batch_size_test=256):
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/files/', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.Grayscale(3),
                                       torchvision.transforms.RandomRotation(15, fill=0),
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])
                                   ),
        batch_size=batch_size_test, shuffle=True)
    return test_loader