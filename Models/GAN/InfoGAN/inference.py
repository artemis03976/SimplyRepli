import torch
import torch.nn.functional as F
from model import Generator
from config.config import InfoGANConfig
from global_utilis import save_and_load, plot


def sample_noise(config, random=True):
    batch_size = config.num_samples
    # sample latent noise
    latent_noise = torch.randn(batch_size, config.latent_noise_dim, device=config.device)

    # sample latent discrete
    if random:
        latent_discrete_idx = torch.randint(
            0, config.latent_discrete_dim, (batch_size, config.num_latent_discrete), device=config.device
        )
    else:
        latent_discrete_idx = torch.arange(
            config.latent_discrete_dim, device=config.device
        ).repeat(batch_size // config.latent_discrete_dim)
    latent_discrete = F.one_hot(latent_discrete_idx, config.latent_discrete_dim).float().view(batch_size, -1)

    # sample latent continuous
    if random:
        latent_continuous = list(torch.rand(batch_size, config.latent_continuous_dim, device=config.device) * 2 - 1)
    else:
        target_continuous = torch.linspace(-1, 1, 10, device=config.device).repeat(batch_size // 10)
        latent_continuous = []
        for i in range(config.latent_continuous_dim):
            latent_c = torch.zeros(batch_size, config.latent_continuous_dim, device=config.device)
            latent_c[:, i] = target_continuous
            latent_continuous.append(latent_c)

    return latent_noise, latent_discrete, latent_discrete_idx, latent_continuous


def inference(config, generator):
    # switch mode
    generator.eval()

    with torch.no_grad():
        latent_noise, latent_discrete, _, latent_continuous = sample_noise(config, random=False)
        # generate for different settings
        for i, lc in enumerate(latent_continuous):
            latent = torch.cat([latent_noise, latent_discrete, lc], dim=1)
            generation = generator(latent)
            out = generation.view(-1, 1, 28, 28)
            # show generated images
            plot.show_img(out, cols=int(config.num_samples ** 0.5))


def main():
    config_path = "config/config.yaml"
    config = InfoGANConfig(config_path)

    generator = Generator(
        config.latent_noise_dim + config.num_latent_discrete * config.latent_discrete_dim + config.latent_continuous_dim,
        config.channel,
        config.feature_size,
        config.base_channel,
        config.num_layers,
    ).to(config.device)

    save_and_load.load_weight(config, generator)

    inference(config, generator)


if __name__ == '__main__':
    main()
