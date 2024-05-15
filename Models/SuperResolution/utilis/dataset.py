from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np


class BSDS500Dataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        super(BSDS500Dataset, self).__init__()

        self.root = os.path.join(root, 'BSDS500\\data\\images', 'train' if train else 'test')
        self.filenames = os.listdir(self.root)
        self.image_filenames = [f for f in self.filenames if f.endswith('.jpg' or '.png')]
        self.train = train

        if transform is not None:
            self.input_transform = transform['input_transform']
            self.target_transform = transform['target_transform']

    def __getitem__(self, index):
        filename = self.image_filenames[index]
        file_path = os.path.join(self.root, filename)
        lr_img, cb, cr = Image.open(file_path).convert('YCbCr').split()
        sr_img = lr_img.copy()

        if self.input_transform is not None:
            lr_img = self.input_transform(lr_img)
        if self.target_transform is not None:
            sr_img = self.target_transform(sr_img)

        if self.train:
            return lr_img, sr_img
        else:
            lr_cb = np.array(cb.resize((lr_img.shape[1], lr_img.shape[2]), Image.BICUBIC))
            lr_cr = np.array(cr.resize((lr_img.shape[1], lr_img.shape[2]), Image.BICUBIC))

            sr_cb = np.array(cb.resize((sr_img.shape[1], sr_img.shape[2]), Image.BICUBIC))
            sr_cr = np.array(cr.resize((sr_img.shape[1], sr_img.shape[2]), Image.BICUBIC))

            return (lr_img, lr_cb, lr_cr), (sr_img, sr_cb, sr_cr)

    def __len__(self):
        return len(self.image_filenames)
