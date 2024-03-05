import torch
import torch.nn.functional as F

from global_utilis import plot, save_and_load
from Models.AutoEncoder.utilis import load_data
from config.config import CVAEConfig
from models.cvae_linear import LinearCVAE
from models.cvae_conv import ConvCVAE


def reconstruction(config, model):
    print("Start reconstruction...")

    test_loader = load_data.load_test_data(config.num_samples)
    images, labels = next(iter(test_loader))
    plot.show_img(images, cols=8)

    with torch.no_grad():
        images = images.to(config.device)
        labels = labels.to(config.device)
        labels = F.one_hot(labels, config.num_classes)
        reconstruction_images, _, _ = model(images, labels)

        if torch.is_tensor(reconstruction_images):
            reconstruction_images = reconstruction_images.detach().cpu().numpy()

    print("End reconstruction...")

    plot.show_img(reconstruction_images, cols=8)


def generation(config, model):
    latent_dim = config.latent_dim_linear if 'linear' in config.network else config.latent_dim_conv

    print("Start generation...")

    with torch.no_grad():
        # sample from standard normal distribution
        latent_z = torch.randn(config.num_samples, latent_dim).to(config.device)
        # randomly generate labels
        labels = torch.randint(0, 10, (config.num_samples,)).to(config.device)
        print(labels)

        labels = F.one_hot(labels, config.num_classes).float()
        # decode
        y = model.label_projection(labels)
        samples = model.decoder(latent_z + y)
        # reshape to image size
        if len(samples.shape) != 4:
            if isinstance(config.img_size, (tuple, list)):
                samples = samples.view(-1, config.channel, config.img_size[0], config.img_size[1])
            else:
                samples = samples.view(-1, config.channel, config.img_size, config.img_size)
        # change back to numpy array
        if torch.is_tensor(samples):
            samples = samples.detach().cpu().numpy()

    print("End generation...")

    plot.show_img(samples, cols=8)


def inference(config, model, reconstruct=False, generate=True):
    if reconstruct:
        reconstruction(config, model)
    if generate:
        generation(config, model)


def main():
    config_path = "config/config.yaml"
    config = CVAEConfig(config_path)

    if config.network == 'cvae_linear':
        if isinstance(config.img_size, (tuple, list)):
            input_dim = output_dim = config.channel * config.img_size[0] * config.img_size[1]
        else:
            input_dim = output_dim = config.img_size ** 2

        model = LinearCVAE(
            input_dim,
            config.latent_dim_linear,
            config.hidden_dims,
            output_dim,
            config.num_classes,
        ).to(config.device)

    elif config.network == 'cvae_conv':
        in_channel = out_channel = config.channel

        model = ConvCVAE(
            in_channel,
            config.latent_dim_conv,
            config.mid_channels,
            out_channel,
            config.img_size,
            config.num_classes,
            config.kernel_size,
        ).to(config.device)

    else:
        raise NotImplementedError(f"Unsupported network: {config.network}")

    save_and_load.load_weight(config, model)

    inference(config, model, reconstruct=True, generate=True)


if __name__ == "__main__":
    main()
