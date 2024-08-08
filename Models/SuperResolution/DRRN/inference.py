import torch
import math
import numpy as np
from global_utilis import plot, save_and_load, save_img
from Models.SuperResolution.utilis import load_data, convert
from config.config import DRRNConfig
from model import DRRN


@torch.no_grad()
def inference(config, model):
    print("Start reconstruction...")

    test_loader = load_data.get_test_loader(config)
    (lr_img, lr_cb, lr_cr), (sr_img, sr_cb, sr_cr) = next(iter(test_loader))

    lr_img_y = lr_img.to(config.device)

    recon_sr_img = model(lr_img_y)

    print("End reconstruction...")

    # convert image from YCbCr to RGB
    lr_imgs = []
    sr_imgs = []

    for y, cb, cr in zip(lr_img, lr_cb, lr_cr):
        rgb_lr_img = convert.convert_ycbcr_to_rgb(y, cb, cr)
        lr_imgs.append(rgb_lr_img)

    for y, cb, cr in zip(sr_img, sr_cb, sr_cr):
        rgb_sr_img = convert.convert_ycbcr_to_rgb(y, cb, cr)
        sr_imgs.append(rgb_sr_img)

    plot.show_img(np.array(lr_imgs).transpose(0, 3, 1, 2), cols=int(math.sqrt(config.num_samples)))
    plot.show_img(np.array(sr_imgs).transpose(0, 3, 1, 2), cols=int(math.sqrt(config.num_samples)))

    save_img.save_img(lr_imgs, "./outputs/LR")
    save_img.save_img(sr_imgs, "./outputs/SR")

    converted_img = []
    for (img_y, img_cb, img_cr) in zip(recon_sr_img, sr_cb, sr_cr):
        img = convert.convert_ycbcr_to_rgb(img_y, img_cb, img_cr)
        converted_img.append(img)

    plot.show_img(np.array(converted_img).transpose(0, 3, 1, 2), cols=int(math.sqrt(config.num_samples)))
    save_img.save_img(converted_img, "./outputs/recon_SR")


def main():
    config_path = "config/config.yaml"
    config = DRRNConfig(config_path)

    model = DRRN(
        config.channel,
        config.mid_channel,
        config.num_layers,
        config.recurse_depth
    ).to(config.device)
    save_and_load.load_weight(config, model)

    inference(config, model)


if __name__ == "__main__":
    main()
