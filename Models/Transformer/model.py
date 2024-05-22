import torch
import torch.nn as nn

from Models.Transformer.modules.encoder import TransformerEncoder
from Models.Transformer.modules.decoder import TransformerDecoder
from Models.Transformer.modules.positional_encoding import PositionalEncoding
from utilis import mask


class Transformer(nn.Module):
    def __init__(
            self,
            src_vocab_size,
            tgt_vocab_size,
            embed_dim,
            ffn_dim,
            num_heads,
            num_layers,
            dropout
    ):
        super(Transformer, self).__init__()

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size

        # token embedding
        self.encoder_embedding = nn.Embedding(src_vocab_size, embed_dim)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, embed_dim)

        # fixed positional encoding
        self.positional_encoding = PositionalEncoding(embed_dim, dropout)

        self.transformer_encoder = TransformerEncoder(embed_dim, num_heads, ffn_dim, num_layers, dropout)
        self.transformer_decoder = TransformerDecoder(embed_dim, num_heads, ffn_dim, num_layers, dropout)

        # projection to the target vocab dim
        self.projection = nn.Linear(embed_dim, tgt_vocab_size)

    def forward(self, src, tgt):
        src_embed = self.positional_encoding(self.encoder_embedding(src))
        tgt_embed = self.positional_encoding(self.decoder_embedding(tgt))

        # get mask for source and target
        src_mask, tgt_mask, cross_mask = mask.get_necessary_mask(src, tgt)

        encoder_output, encoder_attn = self.transformer_encoder(src_embed, src_mask)
        decoder_output, decoder_attn, cross_attn = self.transformer_decoder(tgt_embed, encoder_output, cross_mask, tgt_mask)

        output = self.projection(decoder_output)

        return output, encoder_attn, decoder_attn, cross_attn

    def inference(self, src, sos_token, max_length=40):
        # encode the source sequence
        src_mask = mask.get_padding_mask(src, src)
        src_embed = self.positional_encoding(self.encoder_embedding(src))
        encoder_output, encoder_attn = self.transformer_encoder(src_embed, src_mask)

        # start with the sos token
        decoder_input = torch.tensor([sos_token]).unsqueeze(0).to(src.device)

        while decoder_input.shape[1] < max_length:
            # decode base on the encoder output and the previous decoded token
            _, tgt_mask, cross_mask = mask.get_necessary_mask(src, decoder_input)
            tgt_embed = self.positional_encoding(self.decoder_embedding(decoder_input))
            decoder_output, _, _ = self.transformer_decoder(tgt_embed, encoder_output, cross_mask, tgt_mask)

            decoder_output = self.projection(decoder_output)
            # get the most probable token and add to the seq
            decoded_idx = torch.argmax(decoder_output, dim=-1)
            decoder_input = torch.cat((decoder_input, decoded_idx[:, -1].unsqueeze(0)), dim=1)

        return decoder_input.squeeze(0)
