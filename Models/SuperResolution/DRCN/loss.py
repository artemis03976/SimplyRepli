import torch
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self, inference_depth, alpha, alpha_decay_epoch, beta):
        super(Loss, self).__init__()

        self.inference_depth = inference_depth
        self.alpha = alpha
        self.alpha_decay = self.alpha / alpha_decay_epoch
        self.beta = beta

        self.loss_func = nn.MSELoss()

    def decay_step(self):
        self.alpha = max(0, self.alpha - self.alpha_decay)

    def forward(self, model, recon_sr_img, sr_img, hidden):
        # compute loss
        loss_1 = self.loss_func(recon_sr_img, sr_img)

        loss_2 = 0.0
        for i in range(self.inference_depth):
            loss_2 += self.loss_func(hidden[i], sr_img)

        loss_2 = loss_2 / self.inference_depth

        loss = loss_1 * (1 - self.alpha) + loss_2 * self.alpha + self.beta * model.get_l2_penalty()

        return loss
