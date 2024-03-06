from torch.utils.data import DataLoader
from torchvision import transforms, datasets


transforms = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])


def get_train_loader(config):
    mnist_train = datasets.MNIST(
        root='../../../datas/mnist',
        train=True,
        transform=transforms,
        download=True
    )

    train_loader = DataLoader(mnist_train, batch_size=config.batch_size, shuffle=True)

    return train_loader


def get_test_loader(config):
    mnist_test = datasets.MNIST(
        root='../../../datas/mnist',
        train=False,
        transform=transforms,
        download=True
    )

    test_loader = DataLoader(mnist_test, batch_size=config.batch_size, shuffle=False)

    return test_loader
