from torch.utils.data import DataLoader
from torchvision import transforms, datasets


def get_transform(config):
    transform = transforms.Compose([
        transforms.Resize(config.img_size),
        transforms.CenterCrop(config.img_size),
        transforms.ToTensor(),
    ])

    return transform


def get_train_loader(config):
    train_dataset = datasets.MNIST(
        root='../../../datas/mnist',
        train=True,
        transform=get_transform(config),
        download=True
    )
    if hasattr(config, 'current_train_target'):
        if config.current_train_target == 'model':
            batch_size = config.model_batch_size
        else:
            batch_size = config.prior_batch_size
    else:
        batch_size = config.batch_size

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader


def get_test_loader(config):
    test_dataset = datasets.MNIST(
        root='../../../datas/mnist',
        train=False,
        transform=get_transform(config),
        download=True
    )

    test_loader = DataLoader(test_dataset, batch_size=config.num_samples, shuffle=False)

    return test_loader
