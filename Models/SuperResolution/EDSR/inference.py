import torch
import math
import numpy as np
from global_utilis import plot, save_and_load, save_img
from Models.SuperResolution.utilis import load_data, convert
from config.config import EDSRConfig
from model import EDSR


@torch.no_grad()
def inference(config, model):
    print("Start reconstruction...")

    test_loader = load_data.get_test_loader(config, mode='rgb')
    lr_img, sr_img = next(iter(test_loader))
    lr_img = lr_img.to(config.device)

    recon_sr_img = model(lr_img)

    print("End reconstruction...")

    plot.show_img(lr_img, cols=int(math.sqrt(config.num_samples)))
    plot.show_img(sr_img, cols=int(math.sqrt(config.num_samples)))

    save_img.save_img(lr_img, "./outputs/LR")
    save_img.save_img(sr_img, "./outputs/SR")

    plot.show_img(recon_sr_img, cols=int(math.sqrt(config.num_samples)))
    save_img.save_img(recon_sr_img, "./outputs/recon_SR")


def main():
    config_path = "config/config.yaml"
    config = EDSRConfig(config_path)

    in_channel = out_channel = config.channel

    model = EDSR(
        in_channel,
        out_channel,
        config.mid_channel,
        config.num_blocks,
        config.scale_factor
    ).to(config.device)

    save_and_load.load_weight(config, model)

    inference(config, model)


if __name__ == "__main__":
    main()
