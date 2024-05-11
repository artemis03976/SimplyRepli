import torch
import torch.nn as nn
from modules.patchifier import PatchEmbedding
from modules.transformer_block import TransformerBlock


class VisionTransformer(nn.Module):
    def __init__(
            self,
            in_channel,
            patch_size,
            img_size,
            num_classes,
            num_encoders=6,
            num_heads=8,
            mlp_dim=1024,
            dropout=0.1,
    ):
        super(VisionTransformer, self).__init__()

        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = in_channel * (patch_size ** 2)

        self.patch_embedding = PatchEmbedding(in_channel, patch_size, self.embed_dim, self.num_patches, dropout)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(self.embed_dim, num_heads, mlp_dim, dropout) for _ in range(num_encoders)
        ])

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, num_classes)
        )

    def forward(self, x):
        x = self.patch_embedding(x)

        for transformer_block in self.transformer_blocks:
            x, _ = transformer_block(x)

        x = self.classifier(x[:, 0, :])

        return x
