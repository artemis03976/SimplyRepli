import torch
from model import Generator
from config.config import BEGANConfig
from global_utilis import save_and_load, plot


def inference(config, generator):
    # switch mode
    generator.eval()

    with torch.no_grad():
        # generate fake data
        z = torch.rand(config.num_samples, config.latent_dim, device=config.device) * 2 - 1
        generation = generator(z)
        # rescale to [0, 1]
        out = 0.5 * (generation + 1)
        out = out.clamp(0, 1)
        out = out.view(config.num_samples, config.channel, config.img_size, config.img_size)

    # show generated images
    plot.show_img(out, cols=8)


def main():
    config_path = "config/config.yaml"
    config = BEGANConfig(config_path)

    generator = Generator(
        config.latent_dim,
        config.base_channel,
        config.channel,
        config.num_layers,
        config.img_size,
    ).to(config.device)

    save_and_load.load_weight(config, generator)

    inference(config, generator)


if __name__ == '__main__':
    main()
