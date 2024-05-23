import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from modules.generator import UNetGenerator
from modules.discriminator import PatchGANDiscriminator
from config.config import Pix2PixConfig
from Models.GAN.utilis import load_data
from global_utilis import save_and_load


def train(config, model, train_loader):
    generator, discriminator = model
    # pre-defined loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer_generator = optim.Adam(generator.parameters(), lr=config.generator_lr, betas=(0.5, 0.999))
    optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=config.discriminator_lr, betas=(0.5, 0.999))

    num_epochs = config.epochs

    print("Start training...")

    for epoch in range(num_epochs):
        # set progress bar
        train_info = tqdm(train_loader, unit="batch")
        train_info.set_description(f"Epoch {epoch + 1}/{num_epochs}")

        # main train step
        total_loss = train_step(
            model, config,
            train_info,
            criterion,
            (optimizer_generator, optimizer_discriminator)
        )

        print(
            '\nEpoch [{}/{}], Generator Loss: {:.4f}, Discriminator Loss: {:.4f}'
            .format(epoch + 1, num_epochs, total_loss[0], total_loss[1])
        )

    print("Finish training...")

    save_and_load.save_model(config, model)


def train_step(model, config, train_info, criterion, optimizer):
    # unpacking
    generator, discriminator = model
    generator.train()
    discriminator.train()
    optimizer_generator, optimizer_discriminator = optimizer

    total_loss_g = 0.0
    total_loss_d = 0.0

    for batch_idx, (image_A, image_B) in enumerate(train_info):
        batch_size = image_A.shape[0]
        image_A = image_A.to(config.device)
        image_B = image_B.to(config.device)
        # labels for discriminator
        real_labels = torch.ones(batch_size, 1, 1, 1, device=config.device)
        fake_labels = torch.zeros(batch_size, 1, 1, 1, device=config.device)

        real_A = image_A
        real_B = image_B

        # step1: train discriminator
        for _ in range(config.d_step):
            optimizer_discriminator.zero_grad()

            fake_B = generator(real_A)

            # calculate loss on real data, concat real A for supervised-liked reason
            real_A2B = torch.cat([real_A, real_B], dim=1)
            output_real = discriminator(real_A2B)
            loss_real = criterion(output_real, real_labels.expand_as(output_real))

            # calculate loss on fake data, concat real A for supervised-liked reason
            fake_A2B = torch.cat([real_A, fake_B], dim=1)
            output_fake = discriminator(fake_A2B.detach())
            loss_fake = criterion(output_fake, fake_labels.expand_as(output_fake))

            loss_discriminator = (loss_real + loss_fake) / 2
            loss_discriminator.backward()
            optimizer_discriminator.step()

            total_loss_d += loss_discriminator.item()

        # step2: train generator
        for _ in range(config.g_step):
            optimizer_generator.zero_grad()

            fake_B = generator(real_A)

            fake_A2B = torch.cat([real_B, fake_B], dim=1)
            output_fake = discriminator(fake_A2B)
            loss_gan = criterion(output_fake, real_labels.expand_as(output_fake))
            # add L1 loss
            loss_l1 = F.l1_loss(fake_B, real_B) * config.l1_lambda

            loss_generator = loss_gan + loss_l1
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
    config = Pix2PixConfig(config_path)

    in_channel = out_channel = config.channel

    generator = UNetGenerator(
        in_channel,
        out_channel,
        config.num_blocks_g,
        config.base_channel,
        config.ch_mult,
    ).to(config.device)

    discriminator = PatchGANDiscriminator(
        in_channel + out_channel,
        config.num_layers_d,
        config.base_channel,
    ).to(config.device)

    train_loader = load_data.get_train_loader(config)

    train(config, (generator, discriminator), train_loader)


if __name__ == '__main__':
    main()
