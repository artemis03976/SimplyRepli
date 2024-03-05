from torch.utils.data import DataLoader
from torchvision import transforms, datasets


transforms = transforms.Compose([
        transforms.ToTensor(),
    ])


def load_train_data(batch_size):
    mnist_train = datasets.MNIST(
        root='../../../datas/mnist',
        train=True,
        transform=transforms,
        download=True
    )

    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)

    return train_loader


def load_test_data(batch_size):
    mnist_test = datasets.MNIST(
        root='../../../datas/mnist',
        train=False,
        transform=transforms,
        download=True
    )

    test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)

    return test_loader
