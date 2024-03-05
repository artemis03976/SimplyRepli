import torch
from Models.Transformer.utilis.load_data import *


def get_padding_mask(seq_q, seq_k):
    pad_mask = torch.eq(seq_k, PAD_token).unsqueeze(1)
    pad_mask = pad_mask.expand(pad_mask.shape[0], seq_q.shape[1], seq_k.shape[1])

    return pad_mask


def get_subsequent_mask(seq):
    batch_size, seq_len = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((batch_size, seq_len, seq_len), device=seq.device, dtype=torch.uint8), diagonal=1
    )

    return subsequent_mask


def get_necessary_mask(src_seq, tgt_seq):
    src_padding_mask = get_padding_mask(src_seq, src_seq)
    tgt_padding_mask = get_padding_mask(tgt_seq, tgt_seq)
    cross_padding_mask = get_padding_mask(tgt_seq, src_seq)

    tgt_mask = get_subsequent_mask(tgt_seq)
    tgt_mask = torch.gt(tgt_mask + tgt_padding_mask, 0)

    return src_padding_mask, tgt_mask, cross_padding_mask
