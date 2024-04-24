import torch
import torch.nn.functional as F
from models.cgan_linear import LinearGenerator
from models.cgan_conv import ConvGenerator
from config.config import CGANConfig
from global_utilis import save_and_load, plot


def inference(config, generator):
    generator.eval()

    latent_dim = config.latent_dim_linear if 'linear' in config.network else config.latent_dim_conv

    with torch.no_grad():
        z_img = torch.randn(config.num_samples, latent_dim, device=config.device)
        z_label = torch.randint(0, config.num_classes, (config.num_samples,), device=config.device)
        print(z_label)

        z_label = F.one_hot(z_label, config.num_classes).to(z_img.dtype)

        generation = generator(z_img, z_label)

        out = 0.5 * (generation + 1)
        out = out.clamp(0, 1)
        out = out.view(config.num_samples, config.channel, config.img_size, config.img_size)

    plot.show_img(out, cols=8)


def main():
    config_path = "config/config.yaml"
    config = CGANConfig(config_path)

    if config.network == 'cgan_linear':
        generator = LinearGenerator(
            config.latent_dim_linear,
            config.G_hidden_dims,
            config.output_dim,
            config.num_classes,
            config.proj_dim,
        ).to(config.device)

    elif config.network == 'cgan_conv':
        generator = ConvGenerator(
            config.latent_dim_conv,
            config.G_mid_channels,
            config.out_channel,
            config.num_classes,
            config.proj_dim,
        ).to(config.device)

    else:
        raise NotImplementedError(f'Unsupported network: {config.network}')

    save_and_load.load_weight(config, generator)

    inference(config, generator)


if __name__ == '__main__':
    main()
