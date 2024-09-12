import torch
import torch.nn as nn
from torchvision import models


class GeneratorLoss(nn.Module):
    def __init__(self, device):
        super(GeneratorLoss, self).__init__()

        network = models.vgg19(pretrained=True).to(device)
        # feature after the last relu
        self.network = nn.Sequential(*list(network.features)[:36]).eval()
        for param in self.network.parameters():
            param.requires_grad = False

        self.mse_loss = nn.MSELoss()

    def forward(self, sr, recon_sr, output_fake):
        sr_features = self.network(sr)
        recon_sr_features = self.network(recon_sr)

        mse_loss = self.mse_loss(sr, recon_sr)
        content_loss = self.mse_loss(sr_features, recon_sr_features)
        adversarial_loss = -torch.mean(torch.log(output_fake))

        return mse_loss + 0.006 * content_loss + 1e-3 * adversarial_loss


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super(DiscriminatorLoss, self).__init__()

        self.loss = nn.BCELoss()

    def forward(self, output_real, output_fake):
        real_loss = self.loss(output_real, torch.ones_like(output_real))
        fake_loss = self.loss(output_fake, torch.zeros_like(output_fake))

        return real_loss + fake_loss
