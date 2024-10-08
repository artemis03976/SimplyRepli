import torch
import torch.nn.functional as F
from model import Generator
from config.config import ACGANConfig
from global_utilis import save_and_load, plot


def inference(config, generator):
    # switch mode
    generator.eval()

    with torch.no_grad():
        # generate fake data
        z_img = torch.randn(config.num_samples, config.latent_dim, device=config.device)
        z_label = torch.randint(0, config.num_classes, (config.num_samples,), device=config.device)
        print(z_label)
        # one hot encoding for labels
        z_label = F.one_hot(z_label, config.num_classes).to(z_img.dtype)
        generation = generator(torch.cat([z_img, z_label], dim=1))
        # rescale to [0, 1]
        out = 0.5 * (generation + 1)
        out = out.clamp(0, 1)
        out = out.view(config.num_samples, config.channel, config.img_size, config.img_size)

    # show generated images
    plot.show_img(out, cols=8)


def main():
    config_path = "config/config.yaml"
    config = ACGANConfig(config_path)

    generator = Generator(
        config.latent_dim + config.num_classes,
        config.proj_dim,
        config.G_num_layers,
        config.channel,
        config.img_size
    ).to(config.device)

    save_and_load.load_weight(config, generator)

    inference(config, generator)


if __name__ == '__main__':
    main()
