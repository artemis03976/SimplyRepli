import torch

from global_utilis import plot, save_and_load
from Models.AutoEncoder.utilis import load_data
from config.config import VAEConfig
from Models.AutoEncoder.VanillaVAE.models.vae_linear import LinearVAE
from Models.AutoEncoder.VanillaVAE.models.vae_conv import ConvVAE


def generation(config, model):
    # get latent dim
    latent_dim = config.latent_dim_linear if 'linear' in config.network else config.latent_dim_conv

    print("Start generation...")

    with torch.no_grad():
        # sample from standard normal distribution
        latent_z = torch.randn(config.num_samples, latent_dim).to(config.device)
        # decode
        samples = model.decoder(latent_z)
        # reshape to an image size
        if len(samples.shape) != 4:
            samples = samples.view(-1, config.channel, config.img_size, config.img_size)

    print("End generation...")
    # show generated images
    plot.show_img(samples, cols=8)


def reconstruction(config, model):
    print("Start reconstruction...")

    # load test data for reconstruction
    test_loader = load_data.get_test_loader(config)
    images, labels = next(iter(test_loader))
    # show original images
    plot.show_img(images, cols=8)

    with torch.no_grad():
        images = images.to(config.device)
        reconstruction_images, _, _ = model(images)

    print("End reconstruction...")
    # show reconstructed images
    plot.show_img(reconstruction_images, cols=8)


def inference(config, model, reconstruct=False, generate=True):
    if reconstruct:
        reconstruction(config, model)
    if generate:
        generation(config, model)


def main():
    config_path = "config/config.yaml"
    config = VAEConfig(config_path)

    if config.network == 'vae_linear':
        # get input and output dims
        if isinstance(config.img_size, (tuple, list)):
            input_dim = output_dim = config.channel * config.img_size[0] * config.img_size[1]
        else:
            input_dim = output_dim = config.channel * config.img_size ** 2

        model = LinearVAE(
            input_dim,
            config.latent_dim_linear,
            config.hidden_dims,
            output_dim,
        ).to(config.device)

    elif config.network == 'vae_conv':
        in_channel = out_channel = config.channel

        model = ConvVAE(
            in_channel,
            config.latent_dim_conv,
            config.mid_channels,
            out_channel,
            config.img_size,
            config.kernel_size,
        ).to(config.device)

    else:
        raise NotImplementedError(f"Unsupported network: {config.network}")

    save_and_load.load_weight(config, model)

    inference(config, model, reconstruct=True, generate=True)


if __name__ == "__main__":
    main()
