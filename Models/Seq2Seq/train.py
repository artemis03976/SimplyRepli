import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from models.seq2seq import Seq2SeqRNN, Seq2SeqGRU, Seq2SeqLSTM
from models.seq2seq_attn import Seq2SeqAttnRNN, Seq2SeqAttnGRU, Seq2SeqAttnLSTM
from config.config import Seq2SeqConfig
from Models.Seq2Seq.utilis.load_data import *
from global_utilis.early_stopping import EarlyStopping
from global_utilis import save_and_load


network_mapping = {
    'rnn': Seq2SeqRNN,
    'gru': Seq2SeqGRU,
    'lstm': Seq2SeqLSTM,
    'rnn_attn': Seq2SeqAttnRNN,
    'gru_attn': Seq2SeqAttnGRU,
    'lstm_attn': Seq2SeqAttnLSTM,
}


def train(config, model, train_loader, val_loader):
    # pre-defined loss function and optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    # ignore the padding in the target sequence
    criterion = nn.NLLLoss(ignore_index=PAD_token)

    num_epochs = config.epochs
    # initialize early stopping
    early_stopping = EarlyStopping(patience=3, delta=0.001, mode='min')

    print("Start training...")

    for epoch in range(config.epochs):
        # set progress bar
        train_info = tqdm(train_loader, unit="batch")
        train_info.set_description(f"Epoch {epoch + 1}/{num_epochs}")

        # main train step
        total_loss = train_step(model, config, train_info, criterion, optimizer)
        # main val step
        val_loss = validation(model, config, val_loader)

        print(
            'Epoch [{}/{}], Loss: {:.4f}'
            .format(epoch + 1, config.epochs, total_loss)
        )
        print(
            'Validation Loss: {:.4f}'
            .format(val_loss)
        )

        # check early stopping condition
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Need Early Stopping")
            break

    print("Finish training...")

    save_and_load.save_model(config, model)


def train_step(model, config, train_info, criterion, optimizer):
    total_loss = 0.0
    for batch_idx, (input_seq, target_seq) in enumerate(train_info):
        input_seq = input_seq.to(config.device)
        target_seq = target_seq.to(config.device)

        # enable teacher forcing
        decoder_outputs, attentions = model(input_seq, target_seq, teacher_forcing_ratio=0.5)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_seq.view(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        # set progress bar info
        train_info.set_postfix(loss=loss.item())

    return total_loss / len(train_info)


def validation(model, config, val_loader):
    # switch mode
    model.eval()
    total_loss = 0.0
    # recounting for mean loss
    num_samples = 0
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_token)

    with torch.no_grad():
        for batch_idx, (input_seq, target_seq) in enumerate(val_loader):
            input_seq = input_seq.to(config.device)
            target_seq = target_seq.to(config.device)

            # disable teacher forcing
            decoder_outputs, attentions = model(input_seq, target_seq, teacher_forcing_ratio=0)

            loss = criterion(
                decoder_outputs.view(-1, decoder_outputs.size(-1)),
                target_seq.view(-1)
            )

            total_loss += loss.item()
            num_samples += input_seq.shape[0]

    return total_loss / num_samples


def main():
    config_path = "config/config.yaml"
    config = Seq2SeqConfig(config_path)

    # get data loader and vocab
    input_lang, output_lang, train_loader, val_loader = get_dataloader(config)
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

    train(config, model, train_loader, val_loader)


if __name__ == "__main__":
    main()
