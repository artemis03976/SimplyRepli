import torch
from model import Generator, Discriminator
from config.config import DCGANConfig
from global_utilis import save_and_load, plot


def inference(config, model):
    generator, discriminator = model
    generator.eval()
    discriminator.eval()

    with torch.no_grad():
        generation = generator(torch.randn(config.num_samples, config.latent_dim, device=config.device))
        out = 0.5 * (generation + 1)
        out = out.clamp(0, 1)
        out = out.view(-1, 1, 32, 32)

    plot.show_img(out, cols=8)


def main():
    config_path = "config/config.yaml"
    config = DCGANConfig(config_path)

    generator = Generator(
        config.latent_dim,
        config.G_mid_channels,
        config.out_channel,
    ).to(config.device)

    discriminator = Discriminator(
        config.in_channel,
        config.D_mid_channels,
    ).to(config.device)

    save_and_load.load_weight(config, (generator, discriminator))

    inference(config, (generator, discriminator))


if __name__ == '__main__':
    main()
