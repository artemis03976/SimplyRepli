import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


num_train_samples_ratio = 0.8


class CarvanaDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root

        self.train = train

        if train:
            self.image_path = os.path.join(root, 'train')
            self.mask_path = os.path.join(root, 'train_masks')

        self.image_files = os.listdir(self.image_path)
        self.mask_files = os.listdir(self.mask_path)

        self.transform = transform

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        mask_file = image_file.replace('.jpg', '_mask.gif')

        image = Image.open(os.path.join(self.image_path, image_file))
        mask = Image.open(os.path.join(self.mask_path, mask_file))

        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)

        mask = mask / mask.max()

        return image, mask

    def __len__(self):
        return len(self.image_files)


def get_train_val_loader(config):
    transform = transforms.Compose([
        transforms.Resize((config.img_size, config.img_size)),
        transforms.ToTensor(),
    ])

    dataset = CarvanaDataset(
        root='../../../datas/Carvana/',
        train=True,
        transform=transform
    )

    total_samples = len(dataset)
    num_train_samples = int(total_samples * num_train_samples_ratio)
    num_val_samples = total_samples - num_train_samples
    train_dataset, val_dataset = random_split(dataset, [num_train_samples, num_val_samples])

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.num_samples, shuffle=False)

    return train_loader, val_loader
