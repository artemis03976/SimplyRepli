import torch
import torch.nn as nn
from torchvision import models


class RelativisticAdversarialLoss(nn.Module):
    def __init__(self, mode=None):
        super(RelativisticAdversarialLoss, self).__init__()

        if mode == 'generator':
            self.real_than_fake_label = 0
            self.fake_than_real_label = 1
        elif mode == 'discriminator':
            self.real_than_fake_label = 1
            self.fake_than_real_label = 0
        else:
            raise ValueError('mode must be either "generator" or "discriminator"')

        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, output_real, output_fake):
        E_real = torch.mean(output_real)
        E_fake = torch.mean(output_fake)
        real_than_fake = self.loss(output_real - E_fake, torch.empty_like(output_real).fill_(self.real_than_fake_label))
        fake_than_real = self.loss(output_fake - E_real, torch.empty_like(output_fake).fill_(self.fake_than_real_label))

        return (real_than_fake + fake_than_real) / 2


class GeneratorLoss(nn.Module):
    def __init__(self, device):
        super(GeneratorLoss, self).__init__()

        network = models.vgg19(pretrained=True).to(device)
        # feature before the last relu
        self.network = nn.Sequential(*list(network.features)[:35]).eval()
        for param in self.network.parameters():
            param.requires_grad = False

        self.perceptual_loss = nn.MSELoss()
        self.content_loss = nn.L1Loss()
        self.adversarial_loss = RelativisticAdversarialLoss(mode='generator')

    def forward(self, sr, recon_sr, output_real, output_fake):
        sr_features = self.network(sr)
        recon_sr_features = self.network(recon_sr)
        perceptual_loss = self.perceptual_loss(sr_features, recon_sr_features)

        content_loss = self.content_loss(sr, recon_sr)
        adversarial_loss = self.adversarial_loss(output_real, output_fake)

        return perceptual_loss + 0.01 * content_loss + 5e-3 * adversarial_loss


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super(DiscriminatorLoss, self).__init__()

        self.adversarial_loss = RelativisticAdversarialLoss(mode='discriminator')

    def forward(self, output_real, output_fake):
        return self.adversarial_loss(output_real, output_fake)
