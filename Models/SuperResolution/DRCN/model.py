import torch
import torch.nn as nn


class DRCN(nn.Module):
    def __init__(
            self,
            in_channel,
            inference_depth
    ):
        super(DRCN, self).__init__()

        self.inference_depth = inference_depth

        self.embedding_net = nn.Sequential(
            nn.Conv2d(in_channel, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.inference_net = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.layer_weight = nn.Parameter(torch.full((inference_depth,), 1.0 / inference_depth))

        self.reconstruction_net = nn.ModuleList([
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        ])

    def get_l2_penalty(self):
        l2_penalty = 0
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                l2_penalty += torch.norm(module.weight)
        return l2_penalty

    def forward(self, x):
        identity = x
        x = self.embedding_net(x)

        hidden = []
        for i in range(self.inference_depth):
            x = self.inference_net(x)
            hidden.append(x)

        output = []
        weight_sum = torch.sum(self.layer_weight)
        for i in range(self.inference_depth):
            weight = self.layer_weight[i]
            inf_hidden = hidden[i]

            out = self.reconstruction_net[0](inf_hidden)
            # out = torch.concat([out, res], dim=1)

            hidden[i] = self.reconstruction_net[1](out)
            out = hidden[i] * weight / weight_sum
            output.append(out)

        x = torch.stack(output, dim=0)
        x = torch.sum(x, dim=0) + identity

        return x, hidden
