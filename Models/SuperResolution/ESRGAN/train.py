import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from global_utilis import save_and_load
from Models.SuperResolution.utilis import load_data
from config.config import SRGANConfig
from model import Generator, Discriminator
from loss import GeneratorLoss, DiscriminatorLoss


def train(config, model, train_loader):
    generator, discriminator = model

    criterion_generator = GeneratorLoss(config.device)
    criterion_discriminator = DiscriminatorLoss()

    optimizer_generator = optim.Adam(generator.parameters(), lr=config.generator_lr)
    optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=config.discriminator_lr)

    num_epochs = config.epochs

    print("Start training...")

    for epoch in range(num_epochs):
        # set progress bar
        train_info = tqdm(train_loader, unit="batch")
        train_info.set_description(f"Epoch {epoch + 1}/{num_epochs}")

        total_loss = train_step(
            model,
            config,
            train_info,
            (criterion_generator, criterion_discriminator),
            (optimizer_generator, optimizer_discriminator)
        )

        print(
            '\nEpoch [{}/{}], Generator Loss: {:.4f}, Discriminator Loss: {:.4f},'
            .format(epoch + 1, num_epochs, total_loss[0], total_loss[1])
        )

    print("Finish training...")

    save_and_load.save_model(config, model)


def train_step(model, config, train_info, criterion, optimizer):
    # unpacking
    generator, discriminator = model
    generator.train()
    discriminator.train()
    criterion_generator, criterion_discriminator = criterion
    optimizer_generator, optimizer_discriminator = optimizer

    total_loss_g = 0.0
    total_loss_d = 0.0

    for batch_idx, (lr_img, sr_img) in enumerate(train_info):
        lr_img = lr_img.to(config.device)
        sr_img = sr_img.to(config.device)

        # get fake data
        recon_sr_img = generator(lr_img)

        # step1: train discriminator
        for _ in range(config.d_step):
            optimizer_discriminator.zero_grad()

            output_real = discriminator(sr_img)
            output_fake = discriminator(recon_sr_img.detach())

            loss_discriminator = criterion_discriminator(output_real, output_fake)
            loss_discriminator.backward()

            optimizer_discriminator.step()

            total_loss_d += loss_discriminator.item()

        # step2: train generator
        for _ in range(config.g_step):
            optimizer_generator.zero_grad()

            output_real = discriminator(sr_img)
            output_fake = discriminator(recon_sr_img)

            loss_generator = criterion_generator(sr_img, recon_sr_img, output_real, output_fake)
            loss_generator.backward()

            optimizer_generator.step()

            total_loss_g += loss_generator.item()
        # set progress bar info
        train_info.set_postfix(loss_d=loss_discriminator.item(), loss_g=loss_generator.item())

    return (
        total_loss_g / (len(train_info) * config.g_step),
        total_loss_d / (len(train_info) * config.d_step),
    )


def main():
    config_path = "config/config.yaml"
    config = SRGANConfig(config_path)

    train_loader = load_data.get_train_loader(config, mode='rgb')

    in_channel = out_channel = config.channel

    generator = Generator(
        in_channel,
        out_channel,
        config.base_channel,
        config.growing_channel,
        config.scale_factor,
        config.num_blocks_g,
    ).to(config.device)

    discriminator = Discriminator(
        in_channel,
        config.base_channel,
        config.num_blocks_d,
    ).to(config.device)

    train(config, (generator, discriminator), train_loader)


if __name__ == "__main__":
    main()
