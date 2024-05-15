import os
from PIL import Image
import torch
import numpy as np
from collections.abc import Iterable


def save_img(image_list, path):
    if not os.path.exists(path):
        os.makedirs(path)

    if not isinstance(image_list, Iterable):
        image_list = [image_list]

    for idx, img in enumerate(image_list):
        if not isinstance(img, Image.Image):
            if isinstance(img, torch.Tensor):
                img = img.detach().cpu().numpy().transpose(1, 2, 0)

            img = (img * 255).astype(np.uint8)
            img = Image.fromarray(img)

        img.save(f"{path}/{idx}.png")

    print(f"Saved images to {path}")
