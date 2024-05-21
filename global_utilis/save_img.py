import os
from PIL import Image
import torch
import numpy as np
from collections.abc import Iterable


def save_img(image_list, path):
    # create directory if not exist
    if not os.path.exists(path):
        os.makedirs(path)

    # protection for single image
    if not isinstance(image_list, Iterable):
        image_list = [image_list]

    for idx, img in enumerate(image_list):
        # convert to PIL image
        if not isinstance(img, Image.Image):
            # convert to numpy array
            if isinstance(img, torch.Tensor):
                img = img.detach().cpu().numpy().transpose(1, 2, 0)
            # scale to [0, 255]
            img = (img * 255).astype(np.uint8)
            img = Image.fromarray(img)

        img.save(f"{path}/{idx}.png")

    print(f"Saved images to {path}")
