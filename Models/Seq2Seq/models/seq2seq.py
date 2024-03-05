import torch.nn as nn
import torch.nn.functional as F
import random

from Models.Seq2Seq.models.modules.Encoder import EncoderRNN, EncoderGRU, EncoderLSTM
from Models.Seq2Seq.models.modules.Decoder import DecoderRNN, DecoderGRU, DecoderLSTM
from Models.Seq2Seq.utilis.load_data import *


# base structure for seq2seq model
class Seq2SeqBase(nn.Module):
    def __init__(self, device):
        super(Seq2SeqBase, self).__init__()
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

        for t in range(MAX_LENGTH):
            # decode from target embedding and encoder hidden states
            decoder_output, decoder_hidden, decoder_cell = self.decoder(decoder_input, decoder_hidden, decoder_cell)
            # holding predictions for each token
            decoder_outputs.append(decoder_output)

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

        return decoder_outputs, None


# RNN backbone
class Seq2SeqRNN(Seq2SeqBase):
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
        super(Seq2SeqRNN, self).__init__(device)
        self.encoder = EncoderRNN(
            src_vocab_size,
            emb_dim,
            hidden_dim,
            num_layers,
            encode_dropout,
            bidirectional,
        )
        self.decoder = DecoderRNN(
            tgt_vocab_size,
            emb_dim,
            hidden_dim,
            num_layers,
            decode_dropout,
            bidirectional,
        )


# GRU backbone
class Seq2SeqGRU(Seq2SeqBase):
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
        super(Seq2SeqGRU, self).__init__(device)
        self.encoder = EncoderGRU(
            src_vocab_size,
            emb_dim,
            hidden_dim,
            num_layers,
            encode_dropout,
            bidirectional,
        )
        self.decoder = DecoderGRU(
            tgt_vocab_size,
            emb_dim,
            hidden_dim,
            num_layers,
            decode_dropout,
            bidirectional,
        )


# LSTM backbone
class Seq2SeqLSTM(Seq2SeqBase):
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
        super(Seq2SeqLSTM, self).__init__(device)
        self.encoder = EncoderLSTM(
            src_vocab_size,
            emb_dim,
            hidden_dim,
            num_layers,
            encode_dropout,
            bidirectional,
        )
        self.decoder = DecoderLSTM(
            tgt_vocab_size,
            emb_dim,
            hidden_dim,
            num_layers,
            decode_dropout,
            bidirectional,
        )
