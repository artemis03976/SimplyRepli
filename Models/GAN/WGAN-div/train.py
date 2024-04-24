import torch
import torch.optim as optim
from tqdm import tqdm
from model import Generator, Discriminator
from config.config import WGANdivConfig
from modules import w_divergence
from Models.GAN.utilis import load_data
from global_utilis import save_and_load


def train(config, model, train_loader):
    generator, discriminator = model

    optimizer_generator = optim.Adam(generator.parameters(), lr=config.generator_lr, betas=(0.5, 0.9))
    optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=config.discriminator_lr, betas=(0.5, 0.9))

    num_epochs = config.epochs

    print("Start training...")

    for epoch in range(num_epochs):
        # set progress bar
        train_info = tqdm(train_loader, unit="batch")
        train_info.set_description(f"Epoch {epoch + 1}/{num_epochs}")

        total_loss = train_step((generator, discriminator), config, train_info,
                                (optimizer_generator, optimizer_discriminator))

        print(
            '\nEpoch [{}/{}], Generator Loss: {:.4f}, Discriminator Loss: {:.4f},'
            .format(
                epoch + 1, num_epochs, total_loss[0], total_loss[1]
            )
        )

    print("Finish training...")

    save_and_load.save_model(config, model)


def train_step(model, config, train_info, optimizer):
    generator, discriminator = model
    generator.train()
    discriminator.train()

    optimizer_generator, optimizer_discriminator = optimizer

    total_loss_g = 0.0
    total_loss_d = 0.0

    for batch_idx, (image, _) in enumerate(train_info):
        batch_size = image.shape[0]
        image = image.to(config.device)

        # step1: train discriminator
        for _ in range(config.d_step):
            optimizer_discriminator.zero_grad()

            real_data = image.requires_grad_(True)

            z = torch.randn(batch_size, config.latent_dim, device=config.device)
            fake_data = generator(z)

            output_real = discriminator(real_data).view(-1)
            # do not use .detach() here since grad of fake data is needed at next step
            output_fake = discriminator(fake_data).view(-1)

            loss_grad = w_divergence.get_w_divergence(
                real_data, output_real, fake_data, output_fake, config.device, k=config.k, p=config.p
            )

            loss_discriminator = torch.mean(output_fake) - torch.mean(output_real) + loss_grad
            loss_discriminator.backward()

            optimizer_discriminator.step()

            total_loss_d += loss_discriminator.item()

        # step2: train generator
        for _ in range(config.g_step):
            optimizer_generator.zero_grad()

            fake_data = generator(torch.randn(batch_size, config.latent_dim, device=config.device))

            output_fake = discriminator(fake_data).view(-1)
            loss_generator = -torch.mean(output_fake)
            loss_generator.backward()

            optimizer_generator.step()

            total_loss_g += loss_generator.item()

        train_info.set_postfix(loss_d=loss_discriminator.item(), loss_g=loss_generator.item())

    return (
        total_loss_g / (len(train_info) * config.g_step),
        total_loss_d / (len(train_info) * config.d_step),
    )


def main():
    config_path = "config/config.yaml"
    config = WGANdivConfig(config_path)

    in_channel = out_channel = config.channel

    generator = Generator(
        config.latent_dim,
        config.G_mid_channels,
        out_channel,
        config.img_size,
    ).to(config.device)

    discriminator = Discriminator(
        in_channel,
        config.D_mid_channels,
        config.img_size,
    ).to(config.device)

    train_loader = load_data.get_train_loader(config)

    train(config, (generator, discriminator), train_loader)


if __name__ == '__main__':
    main()
