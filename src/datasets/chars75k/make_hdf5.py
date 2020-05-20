import h5py
import torch
import torchvision
import torchvision.transforms.transforms as transforms
import torchvision.datasets as datasets
import numpy as np

def store_many_hdf5(images, labels, name):
    """ Stores an array of images to HDF5.
        Parameters:
        ---------------
        images       images array, (N, 32, 32, 3) to be stored
        labels       labels array, (N, 1) to be stored
    """
    num_images = len(images)

    # Create a new HDF5 file
    file = h5py.File(f"{name}.h5", "w")

    # Create a dataset in the file
    dataset = file.create_dataset(
        "images", data=images
    )
    meta_set = file.create_dataset(
        "meta", data=labels
    )
    file.close()

def read_many_hdf5(name):
    """ Reads image from HDF5.
        Parameters:
        ---------------
        num_images   number of images to read

        Returns:
        ----------
        images      images array, (N, 32, 32, 3) to be stored
        labels      associated meta data, int label (N, 1)
    """
    images, labels = [], []

    # Open the HDF5 file
    file = h5py.File(f"{name}.h5", "r+")

    images = np.array(file["/images"])
    labels = np.array(file["/meta"])

    return images, labels


def generate_data(data_folder, filename):
    data_folder = 'D:\CharsFont\Bmp'
    transform = {
        'train': transforms.Compose(
            [transforms.Resize([28, 28]),
             transforms.Grayscale(num_output_channels=1),
             transforms.ToTensor(),
             transforms.Normalize((0.1307,), (0.3081,))]),

        'test': transforms.Compose(
            [transforms.Resize([28, 28]),
             transforms.Grayscale(num_output_channels=1),
             transforms.ToTensor(),
             transforms.Normalize((0.1307,), (0.3081,))])
    }
    data = datasets.ImageFolder(root = data_folder, transform=transform['train'])
    data_loader = torch.utils.data.DataLoader(
        data, batch_size=len(data), shuffle=True)

    test_dataset_array = next(iter(data_loader))
    images = test_dataset_array[0].numpy()
    labels = test_dataset_array[1].numpy()

    store_many_hdf5(images, labels, filename)