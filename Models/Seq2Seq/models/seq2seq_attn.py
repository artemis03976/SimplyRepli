import torch.nn as nn
import torch.nn.functional as F
import random

from Models.Seq2Seq.models.modules.Encoder import EncoderRNN, EncoderGRU, EncoderLSTM
from Models.Seq2Seq.models.modules.DecoderAttn import DecoderAttnRNN, DecoderAttnGRU, DecoderAttnLSTM
from Models.Seq2Seq.utilis.load_data import *


# base structure for seq2seq model
class Seq2SeqAttnBase(nn.Module):
    def __init__(self, device):
        super(Seq2SeqAttnBase, self).__init__()
        self.encoder = None
        self.decoder = None
        self.device = device

    def forward(self, input_seq, target_seq, teacher_forcing_ratio=0.5):
        batch_size = input_seq.shape[0]

        # initialize the output sequence
        decoder_outputs = []
        # get the decoder input for the first time, which is SOS token, to signal decode
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long).fill_(SOS_token).to(self.device)

        # encode the input sequence
        encoder_output, encoder_hidden, encoder_cell = self.encoder(input_seq)
        # pass the hidden states to the decoder for decoding
        decoder_hidden, decoder_cell = encoder_hidden, encoder_cell

        attentions = []

        for t in range(MAX_LENGTH):
            # decode from target embedding and encoder hidden states
            decoder_output, decoder_hidden, decoder_cell, attn_weights = self.decoder(decoder_input, decoder_hidden,
                                                                                      decoder_cell, encoder_output)
            # holding predictions for each token
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            # randomly decide using teacher forcing or not
            teacher_forcing = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top_token_val, top_token_idx = decoder_output.topk(1)

            # Teacher forcing: use the target token as the next input
            # Without teacher forcing: use its own predictions as the next input
            decoder_input = target_seq[:, t].unsqueeze(1) if teacher_forcing else top_token_idx.squeeze(-1).detach()

        # cat to be a tensor
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.stack(attentions, dim=1)

        return decoder_outputs, attentions


# RNN backbone
class Seq2SeqAttnRNN(Seq2SeqAttnBase):
    def __init__(
            self,
            src_vocab_size,
            tgt_vocab_size,
            emb_dim,
            hidden_dim,
            num_layers,
            encode_dropout,
            decode_dropout,
            bidirectional,
            device,
    ):
        super(Seq2SeqAttnRNN, self).__init__(device)
        self.encoder = EncoderRNN(
            src_vocab_size,
            emb_dim,
            hidden_dim,
            num_layers,
            encode_dropout,
            bidirectional,
        )
        self.decoder = DecoderAttnRNN(
            tgt_vocab_size,
            emb_dim,
            hidden_dim,
            num_layers,
            decode_dropout,
            bidirectional,
        )


# GRU backbone
class Seq2SeqAttnGRU(Seq2SeqAttnBase):
    def __init__(
            self,
            src_vocab_size,
            tgt_vocab_size,
            emb_dim,
            hidden_dim,
            num_layers,
            encode_dropout,
            decode_dropout,
            bidirectional,
            device,
    ):
        super(Seq2SeqAttnGRU, self).__init__(device)
        self.encoder = EncoderGRU(
            src_vocab_size,
            emb_dim,
            hidden_dim,
            num_layers,
            encode_dropout,
            bidirectional,
        )
        self.decoder = DecoderAttnGRU(
            tgt_vocab_size,
            emb_dim,
            hidden_dim,
            num_layers,
            decode_dropout,
            bidirectional,
        )


# LSTM backbone
class Seq2SeqAttnLSTM(Seq2SeqAttnBase):
    def __init__(
            self,
            src_vocab_size,
            tgt_vocab_size,
            emb_dim,
            hidden_dim,
            num_layers,
            encode_dropout,
            decode_dropout,
            bidirectional,
            device,
    ):
        super(Seq2SeqAttnLSTM, self).__init__(device)
        self.encoder = EncoderLSTM(
            src_vocab_size,
            emb_dim,
            hidden_dim,
            num_layers,
            encode_dropout,
            bidirectional,
        )
        self.decoder = DecoderAttnLSTM(
            tgt_vocab_size,
            emb_dim,
            hidden_dim,
            num_layers,
            decode_dropout,
            bidirectional,
        )
