import torch
import torch.nn as nn
from Models.MultiModal.CLIP.modules.base_transformer import TransformerBlock


class PatchEmbedding(nn.Module):
    def __init__(self, in_channel, patch_size, embed_dim, num_patches):
        super(PatchEmbedding, self).__init__()

        # use conv to patchify
        self.patchifier = nn.Sequential(
            nn.Conv2d(in_channel, embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(start_dim=2),
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim), requires_grad=True)

        self.positional_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim), requires_grad=True)

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)

        x = self.patchifier(x)
        x = x.permute(0, 2, 1)

        x = torch.cat([cls_token, x], dim=1)

        x = self.norm(x + self.positional_embedding)

        return x


class ImageEncoder(nn.Module):
    def __init__(
            self,
            in_channel,
            patch_size,
            img_size,
            align_dim,
            num_layers=6,
            num_heads=8,
            mlp_dim=1024,
    ):
        super(ImageEncoder, self).__init__()

        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = in_channel * (patch_size ** 2)

        self.patch_embedding = PatchEmbedding(in_channel, patch_size, self.embed_dim, self.num_patches)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(self.embed_dim, num_heads, mlp_dim) for _ in range(num_layers)
        ])

        self.align_projection = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, align_dim)
        )

    def forward(self, x):
        mask = None
        x = self.patch_embedding(x)

        for transformer_block in self.transformer_blocks:
            x, _ = transformer_block(x, mask)

        x = self.align_projection(x[:, 0, :])

        return x
