import torch.nn as nn


class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_conditions):
        super(ConditionalBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.num_conditions = num_conditions

        self.bn = nn.BatchNorm2d(num_features, affine=False)

        # conditional embedding to hold extra info
        self.condition_proj = nn.Embedding(num_conditions, num_features * 2)

    def forward(self, x, condition):
        normalized_x = self.bn(x)

        gamma, beta = self.condition_proj(condition).chunk(2, dim=1)
        gamma = gamma.view(-1, self.num_features, 1, 1)
        beta = beta.view(-1, self.num_features, 1, 1)

        out = gamma * normalized_x + beta
        return out
