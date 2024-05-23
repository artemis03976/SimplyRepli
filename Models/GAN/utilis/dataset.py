import os
import random
from PIL import Image
from torch.utils.data import Dataset


class CycleGANDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root_A = os.path.join(root, 'trainA' if train else 'testA')
        self.root_B = os.path.join(root, 'trainB' if train else 'testB')

        self.transform = transform

        self.filenames_A = os.listdir(self.root_A)
        self.filenames_B = os.listdir(self.root_B)

    def __len__(self):
        return max(len(self.filenames_A), len(self.filenames_B))

    def __getitem__(self, idx):
        filename_A = self.filenames_A[idx % len(self.filenames_A)]
        file_path_A = os.path.join(self.root_A, filename_A)

        # random choose image B to avoid fixed pair
        idx_B = random.randint(0, len(self.filenames_B) - 1)
        filename_B = self.filenames_B[idx_B]
        file_path_B = os.path.join(self.root_B, filename_B)

        image_A = Image.open(file_path_A)
        image_B = Image.open(file_path_B)

        # apply transform
        if self.transform is not None:
            image_A = self.transform(image_A)
            image_B = self.transform(image_B)

        return image_A, image_B


class Pix2PixDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = os.path.join(root, 'train' if train else 'test')

        self.transform = transform

        self.filenames = os.listdir(self.root)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        file_path = os.path.join(self.root, filename)

        image = Image.open(file_path)

        # split image A and B from one image
        width, height = image.size
        image_A = image.crop((0, 0, width // 2, height))
        image_B = image.crop((width // 2, 0, width, height))

        # apply transform
        if self.transform is not None:
            image_A = self.transform(image_A)
            image_B = self.transform(image_B)

        return image_A, image_B
