import torch
from torch.utils.data import TensorDataset, random_split, DataLoader
from Models.RNN.utilis.preprocess import *
from Models.RNN.utilis.dataset import IMDBDataset
import numpy as np


MAX_LENGTH = 300  # maximum length of a sentence
num_train_samples_ratio = 0.8


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
    data = IMDBDataset(root=save_dir, train=True)

    vocab = data.vocab
    # transform original text into tensors
    text_tensor = textlist_to_tensor(vocab, data.text, config.device)
    label_tensor = labels_to_tensor(data.labels, config.device)

    # create tensor dataset
    dataset = TensorDataset(text_tensor, label_tensor)

    # split dataset into train and validation
    total_samples = len(dataset)
    num_train_samples = total_samples * num_train_samples_ratio
    num_val_samples = total_samples - num_train_samples
    train_dataset, val_dataset = random_split(dataset, [num_train_samples, num_val_samples])

    # create data loader
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    return vocab, train_loader, val_loader


def get_test_loader(config):
    save_dir = '../../datas/IMDB'
    data = IMDBDataset(root=save_dir, train=False)

    vocab = data.vocab
    # transform original text into tensors
    text_tensor = textlist_to_tensor(vocab, data.text, config.device)
    label_tensor = labels_to_tensor(data.labels, config.device)

    # create data loader
    dataset = TensorDataset(text_tensor, label_tensor)
    test_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    return vocab, test_loader
