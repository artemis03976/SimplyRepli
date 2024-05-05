import torch
import torch.nn as nn
from Models.Diffusion.base_model.unet import UNet


class DDIM(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            num_res_blocks,
            base_channel,
            time_embed_channel,
            ch_mult,
            num_time_step=1000,
            num_sample_step=100,
            betas=(0.0001, 0.02),
            eta=0.0,
    ):
        super(DDIM, self).__init__()

        self.unet = UNet(
            in_channel,
            out_channel,
            num_res_blocks,
            base_channel,
            time_embed_channel,
            ch_mult,
        )

        self.num_time_step = num_time_step
        self.step = num_time_step // num_sample_step
        self.sample_step = list(range(0, num_time_step, self.step))

        beta_sequence = torch.linspace(*betas, num_time_step)

        alphas = 1.0 - beta_sequence
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

        self.eta = eta

    def forward(self, image):
        noise = torch.randn_like(image)

        time_step = torch.randint(0, self.num_time_step, (image.shape[0],), device=image.device)

        alpha_t = self.alphas_cumprod[time_step].view(-1, 1, 1, 1)

        noisy_image = torch.sqrt(alpha_t) * image + torch.sqrt(1 - alpha_t) * noise

        return self.unet(noisy_image, time_step), noise

    @torch.no_grad()
    def inference(self, x_t):
        for t in reversed(self.sample_step):
            noise = torch.randn_like(x_t) if t > 0 else 0.0

            pred_noise = self.unet(x_t, torch.full((x_t.shape[0],), t, device=x_t.device))

            # get alphas
            current_alpha = self.alphas_cumprod[t]
            prev_alpha = self.alphas_cumprod[t - self.step] if t > 0 else torch.tensor(1.0)

            # calculate the predicted x_0
            pred_x_0 = (x_t - torch.sqrt(1 - current_alpha) * pred_noise) / torch.sqrt(current_alpha)

            # calculate sigma
            sigma = self.eta * torch.sqrt((1 - prev_alpha) / (1 - current_alpha) * (1 - current_alpha / prev_alpha))

            # calculate direction pointing
            direction = torch.sqrt(1 - prev_alpha - sigma ** 2) * pred_noise

            # calculate x_(t-1)
            x_t = torch.sqrt(prev_alpha) * pred_x_0 + direction + sigma * noise

        return x_t
