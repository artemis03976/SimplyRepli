from torch.utils.data import DataLoader
from torchvision import transforms, datasets


# TODO: custom dataset


def get_transform(config):
    transform = transforms.Compose([
        transforms.Resize(config.img_size),
        transforms.CenterCrop(config.img_size),
        transforms.ToTensor(),
    ])

    return transform


def get_train_loader(config):
    if config.dataset == 'mnist':
        dataset = datasets.MNIST(
            root='../../../datas/mnist',
            train=True,
            transform=get_transform(config),
            download=True
        )
    elif config.dataset == 'fashion_mnist':
        dataset = datasets.FashionMNIST(
            root='../../../datas/fashion_mnist',
            train=True,
            transform=get_transform(config),
            download=True
        )
    elif config.dataset == 'cifar10':
        dataset = datasets.CIFAR10(
            root="../../../datas/cifar10",
            train=True,
            transform=get_transform(config),
            download=True,
        )
    else:
        raise NotImplementedError('Unsupported dataset: {}'.format(config.dataset))

    # special check for VQVAE prior network
    if hasattr(config, 'current_train_target'):
        if config.current_train_target == 'model':
            batch_size = config.model_batch_size
        else:
            batch_size = config.prior_batch_size
    else:
        batch_size = config.batch_size

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return train_loader


def get_test_loader(config):
    if config.dataset == 'mnist':
        dataset = datasets.MNIST(
            root='../../../datas/mnist',
            train=False,
            transform=get_transform(config),
            download=True
        )
    elif config.dataset == 'fashion_mnist':
        dataset = datasets.FashionMNIST(
            root='../../../datas/fashion_mnist',
            train=False,
            transform=get_transform(config),
            download=True
        )
    elif config.dataset == 'cifar10':
        dataset = datasets.CIFAR10(
            root="../../../datas/cifar10",
            train=False,
            transform=get_transform(config),
            download=True,
        )
    else:
        raise NotImplementedError('Unsupported dataset: {}'.format(config.dataset))

    test_loader = DataLoader(dataset, batch_size=config.num_samples, shuffle=False)

    return test_loader
