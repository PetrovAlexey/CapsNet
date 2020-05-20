import torch
import torchvision

def get_train_valid_loader(batch_size_train, batch_size_val):
    emnist_trainset = torchvision.datasets.EMNIST(root='./files/', train=True, download=True, split='balanced',
                                                  transform=torchvision.transforms.Compose([
                                                      torchvision.transforms.Grayscale(num_output_channels=3),
                                                      torchvision.transforms.ToTensor(),
                                                      torchvision.transforms.Normalize(
                                                          (0.1307,), (0.3081,))
                                                  ]))

    train_size = int(0.9 * len(emnist_trainset))
    test_size = len(emnist_trainset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(emnist_trainset, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size_train, shuffle=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size_val, shuffle=True)

    # train_loader = torch.utils.data.DataLoader(
    #     torchvision.datasets.EMNIST('/files/', train=True, download=True, split='balanced',
    #                                 transform=torchvision.transforms.Compose([
    #                                     torchvision.transforms.Grayscale(num_output_channels=3),
    #                                     torchvision.transforms.ToTensor(),
    #                                     torchvision.transforms.Normalize(
    #                                         (0.1307,), (0.3081,))
    #                                 ])
    #                                 ),
    #     batch_size=batch_size_train, shuffle=True)
    #
    # val_loader = torch.utils.data.DataLoader(
    #     torchvision.datasets.EMNIST('/files/', train=False, download=True, split='balanced',
    #                                 transform=torchvision.transforms.Compose([
    #                                     torchvision.transforms.RandomRotation(0, fill=(0,)),
    #                                     torchvision.transforms.Grayscale(num_output_channels=3),
    #                                     torchvision.transforms.ToTensor(),
    #                                     torchvision.transforms.Normalize(
    #                                         (0.1307,), (0.3081,))
    #                                 ])
    #                                 ),
    #     batch_size=batch_size_val, shuffle=True)
    return train_loader, val_loader


def get_test_loader(batch_size_test=256):
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.EMNIST('/files/', train=False, download=True, split='balanced',
                                    transform=torchvision.transforms.Compose([
                                        torchvision.transforms.RandomRotation(0, fill=(0,)),
                                        torchvision.transforms.Grayscale(num_output_channels=3),
                                        torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(
                                            (0.1307,), (0.3081,))
                                    ])
                                    ),
        batch_size=batch_size_test, shuffle=True)
    return test_loader