from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from Models.GAN.utilis.dataset import Pix2PixDataset, CycleGANDataset


def get_transform(config):
    if not hasattr(config, 'channel') or config.channel == 1:
        if config.network == 'infogan':
            transform = transforms.Compose([
                transforms.Resize(28),
                transforms.CenterCrop(28),
                transforms.ToTensor(),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(32),
                transforms.CenterCrop(32),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
    else:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    return transform


def get_train_loader(config):
    if config.network == 'pix2pix':
        train_dataset = Pix2PixDataset(
            root='../../../datas/pix2pix/cityscapes',
            train=True,
            transform=get_transform(config)
        )

    elif config.network == 'cyclegan':
        train_dataset = CycleGANDataset(
            root='../../../datas/cyclegan/vangogh2photo',
            train=True,
            transform=get_transform(config)
        )

    else:
        train_dataset = datasets.MNIST(
            root='../../../datas/mnist',
            train=True,
            transform=get_transform(config),
            download=True
        )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    return train_loader


def get_test_loader(config):
    if config.network == 'pix2pix':
        test_dataset = Pix2PixDataset(
            root='../../../datas/pix2pix/cityscapes',
            train=False,
            transform=get_transform(config)
        )

    elif config.network == 'cyclegan':
        test_dataset = CycleGANDataset(
            root='../../../datas/cyclegan/vangogh2photo',
            train=False,
            transform=get_transform(config)
        )

    else:
        test_dataset = datasets.MNIST(
            root='../../../datas/mnist',
            train=False,
            transform=get_transform(config),
            download=True
        )

    test_loader = DataLoader(test_dataset, batch_size=config.num_samples, shuffle=False)

    return test_loader
