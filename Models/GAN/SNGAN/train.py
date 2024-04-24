import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import Generator, Discriminator
from config.config import SNGANConfig
from Models.GAN.utilis import load_data
from global_utilis import save_and_load


def train(config, model, train_loader):
    generator, discriminator = model

    criterion = nn.BCEWithLogitsLoss()

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
            '\nEpoch [{}/{}], Generator Loss: {:.4f}, Discriminator Loss: {:.4f}'
            .format(
                epoch + 1, num_epochs, total_loss[0], total_loss[1]
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
    total_loss_d = 0.0

    for batch_idx, (image, _) in enumerate(train_info):
        batch_size = image.shape[0]
        image = image.to(config.device)

        real_labels = torch.ones(batch_size, 1, device=config.device)
        fake_labels = torch.zeros(batch_size, 1, device=config.device)

        # step1: train discriminator
        for _ in range(config.d_step):
            optimizer_discriminator.zero_grad()

            real_data = image

            fake_data = generator(torch.randn(batch_size, config.latent_dim, device=config.device))

            output_real = discriminator(real_data)
            loss_real = criterion(output_real, real_labels)

            output_fake = discriminator(fake_data.detach())
            loss_fake = criterion(output_fake, fake_labels)

            loss_discriminator = loss_real + loss_fake
            loss_discriminator.backward()

            optimizer_discriminator.step()

            total_loss_d += loss_discriminator.item()

        # step2: train generator
        for _ in range(config.g_step):
            optimizer_generator.zero_grad()

            fake_data = generator(torch.randn(batch_size, config.latent_dim, device=config.device))

            output_fake = discriminator(fake_data)
            loss_generator = criterion(output_fake, real_labels)
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
    config = SNGANConfig(config_path)

    in_channel = out_channel = config.channel

    generator = Generator(
        config.latent_dim,
        config.G_mid_channels,
        out_channel,
        config.img_size
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
