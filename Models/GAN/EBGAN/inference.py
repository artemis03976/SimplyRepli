import torch
from model import Generator
from config.config import EBGANConfig
from global_utilis import save_and_load, plot


def inference(config, generator):
    generator.eval()

    with torch.no_grad():
        generation = generator(torch.randn(config.num_samples, config.latent_dim, device=config.device))
        out = 0.5 * (generation + 1)
        out = out.clamp(0, 1)
        out = out.view(config.num_samples, config.channel, config.img_size, config.img_size)

    plot.show_img(out, cols=8)


def main():
    config_path = "config/config.yaml"
    config = EBGANConfig(config_path)

    generator = Generator(
        config.latent_dim,
        config.G_mid_channels,
        config.channel,
        config.img_size,
    ).to(config.device)

    save_and_load.load_weight(config, generator)

    inference(config, generator)


if __name__ == '__main__':
    main()
