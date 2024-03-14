import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import itertools
from modules.generator import ResnetGenerator
from modules.discriminator import PatchGANDiscriminator
from config.config import CycleGANConfig
from Models.GAN.utilis import load_data
from global_utilis import save_and_load


def train(config, model, train_loader):
    generator_A, generator_B, discriminator_A, discriminator_B = model

    optimizer_generator = optim.Adam(
        itertools.chain(generator_A.parameters(), generator_B.parameters()), lr=config.generator_lr, betas=(0.5, 0.999)
    )
    optimizer_discriminator = optim.Adam(
        itertools.chain(discriminator_A.parameters(), discriminator_B.parameters()), lr=config.discriminator_lr, betas=(0.5, 0.999)
    )

    criterion = nn.MSELoss()

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
            criterion,
            (optimizer_generator, optimizer_discriminator)
        )

        print(
            '\nEpoch [{}/{}], Generator Loss: {:.4f}, Discriminator Loss: {:.4f}'
            .format(
                epoch + 1, num_epochs, total_loss[0], total_loss[1]
            )
        )

    print("Finish training...")

    save_and_load.save_model(config, model)


def train_step(model, config, train_info, criterion, optimizer):
    generator_A, generator_B, discriminator_A, discriminator_B = model

    generator_A.train()
    generator_B.train()
    discriminator_A.train()
    discriminator_B.train()

    optimizer_generator, optimizer_discriminator = optimizer

    total_loss_g = 0.0
    total_loss_d = 0.0

    for batch_idx, (image_A, image_B) in enumerate(train_info):
        batch_size = image_A.shape[0]
        image_A = image_A.to(config.device)
        image_B = image_B.to(config.device)

        real_labels = torch.ones(batch_size, 1, 1, 1, device=config.device)
        fake_labels = torch.zeros(batch_size, 1, 1, 1, device=config.device)

        real_A = image_A
        real_B = image_B

        # step1: train discriminator
        for _ in range(config.d_step):
            # backward on discriminator A, which distinguish image B real or not
            discriminator_A.zero_grad()

            fake_B = generator_A(real_A)

            output_real = discriminator_A(real_B)
            loss_real = criterion(output_real, real_labels.expand_as(output_real))

            output_fake = discriminator_A(fake_B.detach())
            loss_fake = criterion(output_fake, fake_labels.expand_as(output_fake))

            loss_discriminator_A = (loss_real + loss_fake) / 2
            loss_discriminator_A.backward()

            # backward on discriminator B, which distinguish image A real or not
            discriminator_B.zero_grad()

            fake_A = generator_B(real_B)

            output_real = discriminator_B(real_A)
            loss_real = criterion(output_real, real_labels.expand_as(output_real))

            output_fake = discriminator_B(fake_A.detach())
            loss_fake = criterion(output_fake, fake_labels.expand_as(output_fake))

            loss_discriminator_B = (loss_real + loss_fake) / 2
            loss_discriminator_B.backward()

            optimizer_discriminator.step()

            loss_discriminator = loss_discriminator_A + loss_discriminator_B

            total_loss_d += loss_discriminator.item()

        # step2: train generator
        for _ in range(config.g_step):
            optimizer_generator.zero_grad()

            fake_B = generator_A(real_A)
            fake_A = generator_B(real_B)

            # calculate gan loss
            output_A = discriminator_A(fake_B)
            output_B = discriminator_B(fake_A)
            loss_gan_A = criterion(output_A, real_labels.expand_as(output_A))
            loss_gan_B = criterion(output_B, real_labels.expand_as(output_B))

            # calculate cycle consistency loss
            recon_A = generator_B(fake_B)
            recon_B = generator_A(fake_A)
            loss_l1_A = F.l1_loss(recon_A, real_A) * config.lambda_A
            loss_l1_B = F.l1_loss(recon_B, real_B) * config.lambda_B

            # calculate identity loss
            identity_A = generator_B(real_A)
            identity_B = generator_A(real_B)
            loss_idt_A = F.l1_loss(identity_A, real_A) * config.lambda_identity
            loss_idt_B = F.l1_loss(identity_B, real_B) * config.lambda_identity

            loss_generator = loss_gan_A + loss_gan_B + loss_l1_A + loss_l1_B + loss_idt_A + loss_idt_B

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
    config = CycleGANConfig(config_path)

    in_channel = out_channel = config.channel

    generator_A = ResnetGenerator(
        in_channel,
        out_channel,
        config.num_blocks_g,
        config.base_channel,
    ).to(config.device)

    generator_B = ResnetGenerator(
        in_channel,
        out_channel,
        config.num_blocks_g,
        config.base_channel,
    ).to(config.device)

    discriminator_A = PatchGANDiscriminator(
        out_channel,
        config.num_layers_d,
        config.base_channel,
    ).to(config.device)

    discriminator_B = PatchGANDiscriminator(
        in_channel,
        config.num_layers_d,
        config.base_channel,
    ).to(config.device)

    train_loader = load_data.get_train_loader(config)

    train(config, (generator_A, generator_B, discriminator_A, discriminator_B), train_loader)


if __name__ == '__main__':
    main()
