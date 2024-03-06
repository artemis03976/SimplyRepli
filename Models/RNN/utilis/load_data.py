import torch
from torch.utils.data import Dataset, TensorDataset, random_split, DataLoader
from Models.RNN.utilis.preprocess import *
import numpy as np


MAX_LENGTH = 300


class IMDBDataset(Dataset):
    def __init__(self, root, train=True):
        if not os.path.exists(os.path.join(root, 'preprocessed')):
            preprocess(root, train=train)

        if train:
            data_path = os.path.join(root, 'preprocessed/train.txt')
        else:
            data_path = os.path.join(root, 'preprocessed/test.txt')

        vocab_path = os.path.join(root, 'preprocessed/vocab.pkl')
        with open(vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)

        with open(data_path, 'r', encoding='utf-8') as f:
            data = [line.strip() for line in f.readlines()]
            self.labels = [int(line[0]) for line in data]
            self.text = [line[2:] for line in data]

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        return self.text[index], self.labels[index]


def sentence_to_tensor(vocab, sentence, device):
    # add PAD token
    input_seq = np.array([PAD_token] * MAX_LENGTH, dtype=np.int32)
    # transform sentence into indexes
    indexes = [vocab[word] if word in vocab.keys() else vocab['UNK'] for word in sentence.split()]
    if len(indexes) > MAX_LENGTH:
        indexes = indexes[:MAX_LENGTH]

    input_seq[0: len(indexes)] = indexes
    return torch.LongTensor(input_seq).to(device).view(1, -1)


def textlist_to_tensor(vocab, textlist, device):
    return torch.cat([sentence_to_tensor(vocab, sentence, device) for sentence in textlist], dim=0)


def labels_to_tensor(labels, device):
    return torch.LongTensor(labels).to(device)


def get_train_val_loader(config):
    save_dir = '../../datas/IMDB'
    data = IMDBDataset(root=save_dir)

    vocab = data.vocab
    text_tensor = textlist_to_tensor(vocab, data.text, config.device)
    label_tensor = labels_to_tensor(data.labels, config.device)

    dataset = TensorDataset(text_tensor, label_tensor)

    total_samples = len(dataset)
    num_val_samples = 5000
    num_train_samples = total_samples - num_val_samples
    train_dataset, val_dataset = random_split(dataset, [num_train_samples, num_val_samples])

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    return vocab, train_loader, val_loader


def get_test_loader(config):
    save_dir = '../../datas/IMDB'
    data = IMDBDataset(root=save_dir, train=False)

    vocab = data.vocab
    text_tensor = textlist_to_tensor(vocab, data.text, config.device)
    label_tensor = labels_to_tensor(data.labels, config.device)

    dataset = TensorDataset(text_tensor, label_tensor)
    test_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    return vocab, test_loader
