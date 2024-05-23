import torch

from modules.generator import ResnetGenerator
from config.config import CycleGANConfig
from Models.GAN.utilis import load_data
from global_utilis import save_and_load, plot


def inference(config, generator, test_loader):
    # switch mode
    generator_A, generator_B = generator
    generator_A.eval()
    generator_B.eval()

    # show original images
    image_A, image_B = next(iter(test_loader))
    plot.show_img(image_A, cols=8)
    plot.show_img(image_B, cols=8)

    image_A = image_A.to(config.device)
    image_B = image_B.to(config.device)

    with torch.no_grad():
        # generate fake images
        fake_B = generator_A(image_A)
        fake_A = generator_B(image_B)
        reconstructed_B = generator_B(fake_A)
        reconstructed_A = generator_A(fake_B)

    # show generated images
    plot.show_img(fake_A, cols=8)
    plot.show_img(fake_B, cols=8)
    # show reconstructed images after cycle
    plot.show_img(reconstructed_A, cols=8)
    plot.show_img(reconstructed_B, cols=8)


def main():
    config_path = "config/config.yaml"
    config = CycleGANConfig(config_path)

    in_channel = out_channel = config.channel

    generator_A = ResnetGenerator(
        in_channel,
        out_channel,
        config.num_blocks_g,
        config.base_channel,
    ).to(config.device)

    generator_B = ResnetGenerator(
        in_channel,
        out_channel,
        config.num_blocks_g,
        config.base_channel,
    ).to(config.device)

    save_and_load.load_weight(config, (generator_A, generator_B))

    test_loader = load_data.get_test_loader(config)

    inference(config, (generator_A, generator_B), test_loader)


if __name__ == '__main__':
    main()
