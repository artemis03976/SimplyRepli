import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from models.cgan_linear import LinearGenerator, LinearDiscriminator
from models.cgan_conv import ConvGenerator, ConvDiscriminator
from config.config import CGANConfig
from Models.GAN.utilis import load_data
from global_utilis import save_and_load


def train(config, model, train_loader):
    generator, discriminator = model

    criterion = nn.BCELoss()

    optimizer_generator = optim.Adam(generator.parameters(), lr=config.generator_lr)
    optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=config.discriminator_lr)

    num_epochs = config.epochs

    print("Start training...")

    for epoch in range(num_epochs):
        # set progress bar
        train_info = tqdm(train_loader, unit="batch")
        train_info.set_description(f"Epoch {epoch + 1}/{num_epochs}")

        total_loss = train_step((generator, discriminator), config, train_info, criterion,
                                (optimizer_generator, optimizer_discriminator))

        print(
            '\nEpoch [{}/{}], Generator Loss: {:.4f}, Discriminator Fake Loss: {:.4f}, Discriminator Real Loss: {:.4f}'
            .format(
                epoch + 1, num_epochs, total_loss[0], total_loss[1], total_loss[2]
            )
        )

    print("Finish training...")

    save_and_load.save_model(config, model)


def train_step(model, config, train_info, criterion, optimizer):
    generator, discriminator = model
    generator.train()
    discriminator.train()

    optimizer_generator, optimizer_discriminator = optimizer

    total_loss_g = 0.0
    total_loss_d_fake = 0.0
    total_loss_d_real = 0.0

    latent_dim = config.latent_dim_linear if 'linear' in config.network else config.latent_dim_conv

    for batch_idx, (image, img_label) in enumerate(train_info):
        batch_size = image.shape[0]
        image = image.to(config.device)
        img_label = img_label.to(config.device)
        img_label = F.one_hot(img_label, config.num_classes).to(image.dtype)

        real_labels = torch.ones(batch_size, 1, device=config.device)
        fake_labels = torch.zeros(batch_size, 1, device=config.device)

        # step1: train discriminator
        for _ in range(config.d_step):
            optimizer_discriminator.zero_grad()

            real_data = image.view(batch_size, -1) if 'linear' in config.network else image

            z_img = torch.randn(batch_size, latent_dim, device=config.device)
            z_label = torch.randint(0, config.num_classes, (batch_size,), device=config.device)
            z_label = F.one_hot(z_label, config.num_classes).to(z_img.dtype)

            fake_data = generator(z_img, z_label)

            output_real = discriminator(real_data, img_label)
            loss_real = criterion(output_real, real_labels)
            loss_real.backward()

            output_fake = discriminator(fake_data.detach(), z_label)
            loss_fake = criterion(output_fake, fake_labels)
            loss_fake.backward()

            optimizer_discriminator.step()

            total_loss_d_fake += loss_fake.item()
            total_loss_d_real += loss_real.item()

        # step2: train generator
        for _ in range(config.g_step):
            optimizer_generator.zero_grad()

            z_img = torch.randn(batch_size, latent_dim, device=config.device)
            z_label = torch.randint(0, config.num_classes, (batch_size,), device=config.device)
            z_label = F.one_hot(z_label, config.num_classes).to(z_img.dtype)

            fake_data = generator(z_img, z_label)

            output_fake = discriminator(fake_data, z_label)
            loss_generator = criterion(output_fake, real_labels)
            loss_generator.backward()

            optimizer_generator.step()

            total_loss_g += loss_generator.item()

        train_info.set_postfix(loss_d_fake=loss_fake.item(), loss_d_real=loss_real.item(), loss_g=loss_generator.item())

    return (
        total_loss_g / (len(train_info) * config.g_step),
        total_loss_d_fake / (len(train_info) * config.d_step),
        total_loss_d_real / (len(train_info) * config.d_step)
    )


def main():
    config_path = "config/config.yaml"
    config = CGANConfig(config_path)

    if config.network == 'cgan_linear':
        input_dim = output_dim = config.img_size * config.img_size

        generator = LinearGenerator(
            config.latent_dim_linear,
            config.G_hidden_dims,
            output_dim,
            config.num_classes,
            config.proj_dim,
        ).to(config.device)

        discriminator = LinearDiscriminator(
            input_dim,
            config.D_hidden_dims,
            config.num_classes,
            config.proj_dim,
        ).to(config.device)

    elif config.network == 'cgan_conv':
        in_channel = out_channel = config.channel

        generator = ConvGenerator(
            config.latent_dim_conv,
            config.G_mid_channels,
            out_channel,
            config.num_classes,
            config.proj_dim,
            config.img_size
        ).to(config.device)

        discriminator = ConvDiscriminator(
            in_channel,
            config.D_mid_channels,
            config.num_classes,
            config.proj_dim,
            config.img_size
        ).to(config.device)

    else:
        raise NotImplementedError(f'Unsupported network: {config.network}')

    train_loader = load_data.get_train_loader(config)

    train(config, (generator, discriminator), train_loader)


if __name__ == '__main__':
    main()
