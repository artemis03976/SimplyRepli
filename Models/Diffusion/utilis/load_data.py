from torch.utils.data import DataLoader
from torchvision import transforms, datasets


def get_transform(config):
    transform = transforms.Compose([
        transforms.Resize(config.img_size),
        transforms.CenterCrop(config.img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]) if config.channel == 1
        else transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    return transform


def get_dataset(dataset_info, config):
    if config.dataset == 'mnist' or None:
        dataset = datasets.MNIST(**dataset_info)
    elif config.dataset == 'fashion_mnist':
        dataset = datasets.FashionMNIST(**dataset_info)
    elif config.dataset == 'cifar10':
        dataset = datasets.CIFAR10(**dataset_info)
    else:
        raise NotImplementedError('Unsupported dataset: {}'.format(config.dataset))

    return dataset


def load_train_data(config):
    root = '../../../datas/' + config.dataset
    dataset_info = {
        'root': root,
        'train': True,
        'transform': get_transform(config),
        'download': True
    }

    train_dataset = get_dataset(dataset_info, config)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    return train_loader


def load_test_data(config):
    root = '../../../datas/' + config.dataset
    dataset_info = {
        'root': root,
        'train': False,
        'transform': get_transform(config),
        'download': True
    }

    test_dataset = get_dataset(dataset_info, config)

    test_loader = DataLoader(test_dataset, batch_size=config.num_samples, shuffle=False)

    return test_loader
