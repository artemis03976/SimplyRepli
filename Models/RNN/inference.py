import torch
from models.rnn import RNNClassifier
from models.lstm import LSTMClassifier
from models.gru import GRUClassifier
from config.config import RNNConfig
from utilis import load_data
from global_utilis import save_and_load


network_mapping = {
    'rnn': RNNClassifier,
    'gru': GRUClassifier,
    'lstm': LSTMClassifier,
}


def inference(config, model, test_loader):
    # switch mode
    model.eval()
    total_accuracy = 0.0
    # recounting for mean accuracy
    num_samples = 0

    with torch.no_grad():
        for batch_idx, (text, label) in enumerate(test_loader):
            text = text.to(config.device)
            label = label.to(config.device)

            output = model(text)

            # calculate accuracy
            total_accuracy += torch.sum(torch.eq(output.argmax(dim=1), label)).item()
            num_samples += text.shape[0]

    accuracy = total_accuracy / num_samples

    print('Test Accuracy: {:.4f}'.format(accuracy))


def main():
    config_path = "config/config.yaml"
    config = RNNConfig(config_path)

    # get test data loader and vocab
    vocab, test_loader = load_data.get_test_loader(config)
    # get vocab size for embedding layer
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

    save_and_load.load_weight(config, model)

    inference(config, model, test_loader)


if __name__ == '__main__':
    main()
