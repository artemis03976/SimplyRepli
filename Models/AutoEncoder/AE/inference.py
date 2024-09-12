import torch
from global_utilis import plot, save_and_load
from Models.AutoEncoder.utilis import load_data
from config.config import AEConfig
from Models.AutoEncoder.AE.models.ae_linear import LinearAE
from Models.AutoEncoder.AE.models.ae_conv import ConvAE


def inference(config, model):
    print("Start reconstruction...")

    # load test data, since AE has no generation ability
    test_loader = load_data.get_test_loader(config)
    images, labels = next(iter(test_loader))
    # show original images
    plot.show_img(images, cols=8)

    with torch.no_grad():
        images = images.to(config.device)
        recon_images = model(images)

    print("End reconstruction...")
    # show reconstructed images
    plot.show_img(recon_images, cols=8)


def main():
    config_path = "config/config.yaml"
    config = AEConfig(config_path)

    if config.network == 'ae_linear':
        # get input and output dims
        if isinstance(config.img_size, (tuple, list)):
            input_dim = output_dim = config.channel * config.img_size[0] * config.img_size[1]
        else:
            input_dim = output_dim = config.channel * config.img_size ** 2

        model = LinearAE(
            input_dim,
            config.latent_dim_linear,
            config.hidden_dims,
            output_dim,
        ).to(config.device)

    elif config.network == 'ae_conv':
        in_channel = out_channel = config.channel

        model = ConvAE(
            in_channel,
            config.latent_dim_conv,
            config.mid_channels,
            out_channel,
            config.img_size,
            config.kernel_size
        ).to(config.device)

    else:
        raise NotImplementedError(f"Unsupported network: {config.network}")

    save_and_load.load_weight(config, model)

    inference(config, model)


if __name__ == "__main__":
    main()
