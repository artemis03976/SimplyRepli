from torch.utils.data import Dataset
from PIL import Image
import os


def load_img(filepath):
    img = Image.open(filepath).convert('YCbCr')
    y, _, _ = img.split()
    return y


class BSDS500Dataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        super(BSDS500Dataset, self).__init__()

        self.root = os.path.join(root, 'BSDS500\\data\\images', 'train' if train else 'test')
        self.filenames = os.listdir(self.root)
        self.image_filenames = [f for f in self.filenames if f.endswith('.jpg' or '.png')]

        if transform is not None:
            self.input_transform = transform['input_transform']
            self.target_transform = transform['target_transform']

    def __getitem__(self, index):
        filename = self.image_filenames[index]
        file_path = os.path.join(self.root, filename)
        lr_img = load_img(file_path)
        sr_img = lr_img.copy()

        if self.input_transform is not None:
            lr_img = self.input_transform(lr_img)
        if self.target_transform is not None:
            sr_img = self.target_transform(sr_img)

        return lr_img, sr_img

    def __len__(self):
        return len(self.image_filenames)
