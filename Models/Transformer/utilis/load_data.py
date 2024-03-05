import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, random_split
from Models.Transformer.utilis.create_vocab_dict import *


# transform sentence into indexes
def sentence_to_indexes(lang, sentence):
    if lang.name == "cmn":
        return [lang.word2index[word] for word in jieba.cut(sentence)]
    else:
        return [lang.word2index[word] for word in sentence.split(' ')]


# transform sentence into tensor based on the index list
def sentence_to_tensor(lang, sentence, device):
    # add PAD token
    input_seq = np.array([PAD_token] * MAX_LENGTH, dtype=np.int32)
    # transform sentence into indexes
    indexes = sentence_to_indexes(lang, sentence)
    # add EOS token
    indexes.append(EOS_token)
    # insert into sequence
    input_seq[0] = 0
    input_seq[1: len(indexes) + 1] = indexes
    return torch.LongTensor(input_seq).to(device).view(1, -1)


# transform sentence pair into tensor
def pair_to_tensor(input_lang, output_lang, pair, device):
    input_tensor = sentence_to_tensor(input_lang, pair[0], device)
    target_tensor = sentence_to_tensor(output_lang, pair[1], device)
    return input_tensor, target_tensor


def get_dataloader(config):
    input_lang, output_lang, pairs = prepare_data(lang1=config.lang1, lang2=config.lang2, reverse=config.reverse)

    input_idxes = []
    target_idxes = []

    for idx, pair in enumerate(pairs):
        input_tensor, target_tensor = pair_to_tensor(input_lang, output_lang, pair, config.device)
        input_idxes.append(input_tensor)
        target_idxes.append(target_tensor)

    dataset = TensorDataset(torch.cat(input_idxes, dim=0), torch.cat(target_idxes, dim=0))

    total_samples = len(dataset)
    num_val_samples = 3000
    num_train_samples = total_samples - num_val_samples
    train_dataset, val_dataset = random_split(dataset, [num_train_samples, num_val_samples])

    train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=config.batch_size)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    return input_lang, output_lang, train_loader, val_loader
