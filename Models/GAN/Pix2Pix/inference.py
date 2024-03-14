import torch

from modules.generator import UNetGenerator
from config.config import Pix2PixConfig
from Models.GAN.utilis import load_data
from global_utilis import save_and_load, plot


def inference(config, generator, test_loader):
    generator.eval()

    image_A, image_B = next(iter(test_loader))
    plot.show_img(image_A, cols=2)
    plot.show_img(image_B, cols=2)

    image_A = image_A.to(config.device)

    with torch.no_grad():
        fake_B = generator(image_A)

    plot.show_img(fake_B, cols=2)


def main():
    config_path = "config/config.yaml"
    config = Pix2PixConfig(config_path)

    in_channel = out_channel = config.channel

    generator = UNetGenerator(
        in_channel,
        out_channel,
        config.num_blocks_g,
        config.base_channel,
        config.ch_mult,
    ).to(config.device)

    save_and_load.load_weight(config, generator)

    test_loader = load_data.get_test_loader(config)

    inference(config, generator, test_loader)


if __name__ == '__main__':
    main()
