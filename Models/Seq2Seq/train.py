import torch.nn as nn
import torch.optim as optim

from models.seq2seq import Seq2SeqRNN, Seq2SeqGRU, Seq2SeqLSTM
from models.seq2seq_attn import Seq2SeqAttnRNN, Seq2SeqAttnGRU, Seq2SeqAttnLSTM
from config.config import Seq2SeqConfig
from Models.Seq2Seq.utilis.load_data import *
from global_utilis import save_and_load


network_mapping = {
    'rnn': Seq2SeqRNN,
    'gru': Seq2SeqGRU,
    'lstm': Seq2SeqLSTM,
    'rnn_attn': Seq2SeqAttnRNN,
    'gru_attn': Seq2SeqAttnGRU,
    'lstm_attn': Seq2SeqAttnLSTM,
}


def train(config, model, train_dataloader):
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    # ignore the padding in the target sequence
    criterion = nn.NLLLoss(ignore_index=PAD_token)

    print("Start training...")

    for epoch in range(config.epochs):
        total_loss = 0
        for batch_idx, data in enumerate(train_dataloader):
            optimizer.zero_grad()

            input_seq, target_seq = data

            decoder_outputs, attentions = model(input_seq, target_seq, teacher_forcing_ratio=0.5)

            loss = criterion(
                decoder_outputs.view(-1, decoder_outputs.size(-1)),
                target_seq.view(-1)
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, config.epochs, total_loss / len(train_dataloader)))

    print("Finish training...")

    save_and_load.save_model(config, model)


def main():
    config_path = "config/config.yaml"
    config = Seq2SeqConfig(config_path)

    input_lang, output_lang, train_loader = get_dataloader(config.batch_size, config.device)

    src_vocab_size = input_lang.n_words
    tgt_vocab_size = output_lang.n_words

    if config.network not in network_mapping:
        raise NotImplementedError(f"Unsupported network: {config.network}")

    model = network_mapping[config.network](
        src_vocab_size,
        tgt_vocab_size,
        config.embed_dim,
        config.hidden_dim,
        config.num_layers,
        config.encode_dropout,
        config.decode_dropout,
        config.bidirectional,
        config.device,
    ).to(config.device)

    train(config, model, train_loader)


if __name__ == "__main__":
    main()
