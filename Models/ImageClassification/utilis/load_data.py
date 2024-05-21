from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets


num_train_samples_ratio = 0.9


def get_transforms(config):
    if config.channel == 1:
        mean = [0.5]
        std = [0.5]
    else:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

    transform = {
        # train dataset with augmentation
        'train': transforms.Compose([
            transforms.Resize(config.img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]),
        'val': transforms.Compose([
            transforms.Resize(config.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]),
        'test': transforms.Compose([
            transforms.Resize(config.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]),
    }

    return transform


# TODO:custom dataset


def get_train_val_loader(config):
    if config.dataset == 'cifar100':
        dataset = datasets.CIFAR100(
            root="../../../datas/cifar100",
            train=True,
            download=True,
        )
    elif config.dataset == 'cifar10':
        dataset = datasets.CIFAR10(
            root="../../../datas/cifar10",
            train=True,
            download=True,
        )
    elif config.dataset == 'mnist':
        dataset = datasets.MNIST(
            root="../../../datas/mnist",
            train=True,
            download=True,
        )
    else:
        raise NotImplementedError('Unsupported dataset: {}'.format(config.dataset))

    total_samples = len(dataset)
    num_train_samples = int(total_samples * num_train_samples_ratio)
    num_val_samples = total_samples - num_train_samples
    train_dataset, val_dataset = random_split(dataset, [num_train_samples, num_val_samples])

    data_transforms = get_transforms(config)
    train_dataset.dataset.transform = data_transforms['train']
    val_dataset.dataset.transform = data_transforms['val']

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    return train_loader, val_loader


def get_test_loader(config):
    if config.dataset == 'cifar100':
        dataset = datasets.CIFAR100(
            root="../../../datas/cifar100",
            train=False,
            transform=get_transforms(config)['test'],
            download=True,
        )
    elif config.dataset == 'cifar10':
        dataset = datasets.CIFAR10(
            root="../../../datas/cifar10",
            train=False,
            transform=get_transforms(config)['test'],
            download=True,
        )
    elif config.dataset == 'mnist':
        dataset = datasets.MNIST(
            root="../../../datas/mnist",
            train=False,
            transform=get_transforms(config)['test'],
            download=True,
        )
    else:
        raise NotImplementedError('Unsupported dataset: {}'.format(config.dataset))

    test_loader = DataLoader(dataset, batch_size=config.num_samples, shuffle=False)

    return test_loader
