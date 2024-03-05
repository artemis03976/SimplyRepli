import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from model import Transformer
from config.config import TransformerConfig
from utilis import load_data
from global_utilis.early_stopping import EarlyStopping
from global_utilis import save_and_load


def modify_tgt_seq(tgt_seq):
    padding = torch.full((tgt_seq.size(0), 1), load_data.PAD_token, device=tgt_seq.device)
    shifted_tgt_seq = torch.cat([tgt_seq[:, 1:], padding], dim=1)

    return shifted_tgt_seq


def train(config, model, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss(ignore_index=load_data.PAD_token)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    num_epochs = config.epochs

    early_stopping = EarlyStopping(patience=3, delta=0.001, mode='min')

    print("Start training...")

    for epoch in range(num_epochs):
        # set progress bar
        train_info = tqdm(train_loader, unit="batch")
        train_info.set_description(f"Epoch {epoch + 1}/{num_epochs}")

        total_loss = train_step(model, config, train_info, criterion, optimizer)

        val_loss = validation(model, config, val_loader)

        early_stopping(val_loss)

        print('\nEpoch [{}/{}], Average Loss: {:.4f}'.format(epoch + 1, num_epochs, total_loss / len(train_loader)))
        print('Validation Accuracy: {:.4f}'.format(val_loss))

        if early_stopping.early_stop:
            print("Need Early Stopping")
            break

    print("Finish training...")

    save_and_load.save_model(config, model)


def train_step(model, config, train_info, criterion, optimizer):
    model.train()
    total_loss = 0.0

    for batch_idx, (src_seq, tgt_seq) in enumerate(train_info):
        src_seq = src_seq.to(config.device)
        tgt_seq = tgt_seq.to(config.device)

        output, encoder_attn, decoder_attn, cross_attn = model(src_seq, tgt_seq)

        # throw out SOS token when calculating loss
        shifted_tgt_seq = modify_tgt_seq(tgt_seq)

        loss = criterion(output.view(-1, model.tgt_vocab_size), shifted_tgt_seq.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        train_info.set_postfix(Loss=loss.item())

    return total_loss


def validation(model, config, val_loader):
    model.eval()
    total_loss = 0.0
    num_samples = 0
    criterion = nn.CrossEntropyLoss(ignore_index=load_data.PAD_token)

    with torch.no_grad():
        for batch_idx, (src_seq, tgt_seq) in enumerate(val_loader):
            src_seq = src_seq.to(config.device)
            tgt_seq = tgt_seq.to(config.device)

            output, encoder_attn, decoder_attn, cross_attn = model(src_seq, tgt_seq)

            # throw out SOS token when calculating loss
            shifted_tgt_seq = modify_tgt_seq(tgt_seq)

            loss = criterion(output.view(-1, model.tgt_vocab_size), shifted_tgt_seq.view(-1))

            total_loss += loss.item()
            num_samples += src_seq.shape[0]

    return total_loss / num_samples


def main():
    config_path = "config/config.yaml"
    config = TransformerConfig(config_path)

    input_lang, output_lang, train_loader, val_loader = load_data.get_dataloader(config)

    src_vocab_size = input_lang.n_words
    tgt_vocab_size = output_lang.n_words

    model = Transformer(
        src_vocab_size,
        tgt_vocab_size,
        config.embed_dim,
        config.ffn_dim,
        config.num_heads,
        config.num_layers,
        config.dropout,
    ).to(config.device)

    train(config, model, train_loader, val_loader)


if __name__ == "__main__":
    main()
