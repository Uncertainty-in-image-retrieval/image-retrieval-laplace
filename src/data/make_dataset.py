from torchvision import datasets, transforms

def get_data(data_dir, transform=None):

    training_data = datasets.KMNIST(data_dir, train=True, download=True, transform=transform)
    test_data = datasets.KMNIST(data_dir, train=False, transform=transform)
    validation_data = datasets.KMNIST(data_dir, train=False, transform=transform)

    return training_data, test_data, validation_data