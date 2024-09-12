import torch
import torch.nn as nn
import numpy as np
from modules.image_encoder import ImageEncoder
from modules.text_encoder import TextEncoder


class CLIP(nn.Module):
    def __init__(
            self,
            align_dim,
            # vision part
            in_channel,
            patch_size,
            img_size,
            vision_num_layers,
            vision_num_heads,
            vision_mlp_dim,
            # text part
            vocab_size,
            text_max_len,
            text_embed_dim,
            text_num_layers,
            text_num_heads,
            text_mlp_dim,
    ):
        super(CLIP, self).__init__()

        self.vision_encoder = ImageEncoder(
            in_channel,
            patch_size,
            img_size,
            align_dim,
            vision_num_layers,
            vision_num_heads,
            vision_mlp_dim,
        )

        self.text_encoder = TextEncoder(
            vocab_size,
            text_max_len,
            text_embed_dim,
            align_dim,
            text_num_layers,
            text_num_heads,
            text_mlp_dim,
        )

        self.scale_factor = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, image, text):
        image_feature = self.vision_encoder(image)
        text_feature = self.text_encoder(text)

        image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
        text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)

        scale_factor = torch.exp(self.scale_factor)

        # calculate cosine similarity
        logits_per_image = scale_factor * torch.matmul(image_feature, text_feature.transpose(0, 1))
        logits_per_text = scale_factor * torch.matmul(text_feature, image_feature.transpose(0, 1))

        return logits_per_image, logits_per_text
