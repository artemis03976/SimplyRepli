from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets


num_train_samples_ratio = 0.9


def get_transforms(config):
    if config.channel == 1:
        mean = [0.5]
        std = [0.5]
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    transform = {
        # train dataset with augmentation
        'train': transforms.Compose([
            transforms.RandomResizedCrop(config.img_size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(config.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(config.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]),
    }

    return transform


# TODO: custom dataset
def get_dataset(dataset_info, config):
    if config.dataset == 'cifar100':
        dataset = datasets.CIFAR100(**dataset_info)
    elif config.dataset == 'cifar10':
        dataset = datasets.CIFAR10(**dataset_info)
    elif config.dataset == 'mnist':
        dataset = datasets.MNIST(**dataset_info)
    elif config.dataset == 'mini-imagenet':
        if dataset_info['train']:
            dataset = datasets.ImageFolder(root=dataset_info['root'] + '/train')
        else:
            dataset = datasets.ImageFolder(root=dataset_info['root'] + '/val')
    else:
        raise NotImplementedError('Unsupported dataset: {}'.format(config.dataset))

    return dataset


def get_train_val_loader(config):
    root = '../../../datas/' + config.dataset
    dataset_info = {
        'root': root,
        'train': True,
        'download': True
    }

    train_dataset = get_dataset(dataset_info, config)

    # split dataset into train and validation
    total_samples = len(train_dataset)
    num_train_samples = int(total_samples * num_train_samples_ratio)
    num_val_samples = total_samples - num_train_samples
    train_dataset, val_dataset = random_split(train_dataset, [num_train_samples, num_val_samples])

    # apply transforms
    data_transforms = get_transforms(config)
    train_dataset.dataset.transform = data_transforms['train']
    val_dataset.dataset.transform = data_transforms['val']

    # create data loader
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    return train_loader, val_loader


def get_test_loader(config):
    root = '../../../datas/' + config.dataset
    dataset_info = {
        'root': root,
        'train': False,
        'transform': get_transforms(config)['test'],
        'download': True
    }

    test_dataset = get_dataset(dataset_info, config)

    test_loader = DataLoader(test_dataset, batch_size=config.num_samples, shuffle=False)

    return test_loader
