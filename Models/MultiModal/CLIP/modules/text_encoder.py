import torch
import torch.nn as nn
from Models.MultiModal.CLIP.modules.base_transformer import TransformerBlock


class TextEncoder(nn.Module):
    def __init__(
            self,
            vocab_size,
            text_max_len,
            embed_dim,
            align_dim,
            num_layers,
            num_heads,
            mlp_dim,
    ):
        super(TextEncoder, self).__init__()

        self.PAD_token = 0

        self.text_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_embedding = nn.Parameter(torch.randn(1, text_max_len, embed_dim), requires_grad=True)

        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_dim)
            for _ in range(num_layers)
        ])

        self.align_projection = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, align_dim)
        )

    def get_padding_mask(self, seq):
        pad_mask = torch.eq(seq, self.PAD_token).unsqueeze(1)
        pad_mask = pad_mask.expand(pad_mask.shape[0], seq.shape[1], seq.shape[1])

        return pad_mask

    def forward(self, input_seq):
        seq = self.text_embedding(input_seq) + self.positional_embedding
        mask = self.get_padding_mask(input_seq)

        for layer in self.layers:
            seq, _ = layer(seq, mask)

        # use eot token
        seq = seq[torch.arange(seq.shape[0]), input_seq.argmax(dim=-1)]
        output = self.align_projection(seq)

        return output
