from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
from Models.SuperResolution.utilis.dataset import BSDS500Dataset


model_with_upscale = ['espcn', 'edsr', 'srgan', 'esrgan']


def get_transform(config):
    if config.network in model_with_upscale:
        input_transform = transforms.Compose([
            transforms.CenterCrop(config.img_size),
            transforms.Resize(config.img_size // config.scale_factor, interpolation=Image.BICUBIC),
            transforms.ToTensor()
        ])
    else:
        input_transform = transforms.Compose([
            transforms.CenterCrop(config.img_size),
            transforms.Resize(config.img_size // config.scale_factor, interpolation=Image.BICUBIC),
            transforms.Resize(config.img_size, interpolation=Image.BICUBIC),
            transforms.ToTensor()
        ])

    target_transform = transforms.Compose([
        transforms.CenterCrop(config.img_size),
        transforms.ToTensor()
    ])

    transform = {
        'input_transform': input_transform,
        'target_transform': target_transform
    }

    return transform


def get_train_loader(config, mode='ycbcr'):
    train_dataset = BSDS500Dataset(
        root='../../../datas/SR-dataset/BSDS500',
        train=True,
        transform=get_transform(config),
        mode=mode
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    return train_loader


def get_test_loader(config, mode='ycbcr'):
    test_dataset = BSDS500Dataset(
        root='../../../datas/SR-dataset/BSDS500',
        train=False,
        transform=get_transform(config),
        mode=mode
    )

    test_loader = DataLoader(test_dataset, batch_size=config.num_samples, shuffle=False)

    return test_loader
