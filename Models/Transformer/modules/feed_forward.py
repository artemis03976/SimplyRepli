import torch
import torch.nn as nn


class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.5):
        super(FeedForwardNetwork, self).__init__()
        self.fc_1 = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.fc_2 = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.fc_1(x)
        x = self.fc_2(x)

        return x
