import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, in_channel, patch_size, embed_dim, num_patches, dropout):
        super(PatchEmbedding, self).__init__()

        # use conv to patchify
        self.patchifier = nn.Sequential(
            nn.Conv2d(in_channel, embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(start_dim=2),
        )

        # class token embedding with random initialize
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim), requires_grad=True)

        # positional embedding with random initialize
        self.positional_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim), requires_grad=True)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # expand class token to fit the input shape
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)

        x = self.patchifier(x)
        # move channel to the last dimension
        x = x.permute(0, 2, 1)

        # add class token to the beginning of the sequence
        x = torch.cat([cls_token, x], dim=1)

        x = x + self.positional_embedding

        x = self.dropout(x)

        return x
