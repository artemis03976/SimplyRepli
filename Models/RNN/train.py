import torch
import torch.nn as nn
import torch.optim as optim
from models.rnn import RNNClassifier
from models.lstm import LSTMClassifier
from models.gru import GRUClassifier
from config.config import RNNConfig
from utilis import load_data
from global_utilis.early_stopping import EarlyStopping
from global_utilis import save_and_load
from tqdm import tqdm


network_mapping = {
    'rnn': RNNClassifier,
    'gru': GRUClassifier,
    'lstm': LSTMClassifier,
}


def train(config, model, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    num_epochs = config.epochs

    early_stopping = EarlyStopping(patience=3, delta=0.001, mode='max')

    print("Start training...")

    for epoch in range(num_epochs):
        # set progress bar
        train_info = tqdm(train_loader, unit="batch")
        train_info.set_description(f"Epoch {epoch + 1}/{num_epochs}")

        total_loss = train_step(model, config, train_info, criterion, optimizer)

        total_accuracy = validation(model, config, val_loader)

        early_stopping(total_accuracy)

        print('\nEpoch [{}/{}], Average Loss: {:.4f}'.format(epoch + 1, num_epochs, total_loss / len(train_loader)))
        print('Validation Accuracy: {:.4f}'.format(total_accuracy))

        if early_stopping.early_stop:
            print("Need Early Stopping")
            break

    print("Finish training...")

    save_and_load.save_model(config, model)


def train_step(model, config, train_info, criterion, optimizer):
    model.train()
    total_loss = 0.0

    for batch_idx, (text, label) in enumerate(train_info):
        text = text.to(config.device)
        label = label.to(config.device)

        output = model(text)

        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        train_info.set_postfix(Loss=loss.item())

    return total_loss


def validation(model, config, val_loader):
    model.eval()
    total_accuracy = 0.0
    num_samples = 0

    with torch.no_grad():
        for batch_idx, (text, label) in enumerate(val_loader):
            text = text.to(config.device)
            label = label.to(config.device)

            output = model(text)

            total_accuracy += torch.sum(torch.eq(output.argmax(dim=1), label)).item()
            num_samples += text.shape[0]

    return total_accuracy / num_samples


def main():
    config_path = "config/config.yaml"
    config = RNNConfig(config_path)

    vocab, train_loader, val_loader = load_data.get_train_val_loader(config)

    src_vocab_size = len(vocab)

    model = network_mapping[config.network](
        src_vocab_size,
        config.embed_dim,
        config.hidden_dim,
        config.num_classes,
        config.num_layers,
        config.bidirectional,
        config.dropout,
    ).to(config.device)

    train(config, model, train_loader, val_loader)


if __name__ == '__main__':
    main()
