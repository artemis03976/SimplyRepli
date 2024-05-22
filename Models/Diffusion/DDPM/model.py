import torch
import torch.nn as nn
from Models.Diffusion.base_model.unet import UNet


class DDPM(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            num_res_blocks,
            base_channel,
            time_embed_channel,
            ch_mult,
            num_time_step=1000,
            betas=(0.0001, 0.02)
    ):
        super(DDPM, self).__init__()

        # noise prediction network
        self.unet = UNet(
            in_channel,
            out_channel,
            num_res_blocks,
            base_channel,
            time_embed_channel,
            ch_mult,
        )

        self.num_time_step = num_time_step

        beta_sequence = torch.linspace(*betas, self.num_time_step)
        self.register_buffer("betas", beta_sequence)

        # calculate alphas and alphas cumprod
        alphas = 1.0 - beta_sequence
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def forward(self, image):
        # sample noises
        noise = torch.randn_like(image)
        # random generate time step
        time_step = torch.randint(0, self.num_time_step, (image.shape[0],), device=image.device)
        # get corresponding alphas
        alpha_t = self.alphas_cumprod[time_step].view(-1, 1, 1, 1)
        # add noise
        noisy_image = torch.sqrt(alpha_t) * image + torch.sqrt(1 - alpha_t) * noise

        return self.unet(noisy_image, time_step), noise

    @torch.no_grad()
    def inference(self, x_t):
        # sample across whole time step
        for t in reversed(range(self.num_time_step)):
            noise = torch.randn_like(x_t) if t > 0 else 0.0
            # get predicted noise
            pred_noise = self.unet(x_t, torch.full((x_t.shape[0],), t, device=x_t.device))
            # calculate mean
            mean = x_t - (self.betas[t] / torch.sqrt(1 - self.alphas_cumprod[t])) * pred_noise
            mean /= torch.sqrt(self.alphas[t])
            # calculate variance
            prev_alpha_cumprod = self.alphas_cumprod[t - 1] if t > 0 else torch.tensor(1.0)
            var = self.betas[t] * (1 - prev_alpha_cumprod) / (1 - self.alphas_cumprod[t])
            var = torch.sqrt(var)
            # calculate x_(t-1)
            x_t = mean + var * noise

        return x_t
