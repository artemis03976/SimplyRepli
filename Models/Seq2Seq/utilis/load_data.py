import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from Models.Seq2Seq.utilis.create_vocab_dict import *


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
    input_seq[:len(indexes)] = indexes
    return torch.LongTensor(input_seq).to(device).view(1, -1)


# transform sentence pair into tensor
def pair_to_tensor(input_lang, output_lang, pair, device):
    input_tensor = sentence_to_tensor(input_lang, pair[0], device)
    target_tensor = sentence_to_tensor(output_lang, pair[1], device)
    return input_tensor, target_tensor


def get_dataloader(batch_size, device):
    input_lang, output_lang, pairs = prepare_data(lang1='eng', lang2='cmn', reverse=True)

    input_idxes = []
    target_idxes = []

    for idx, pair in enumerate(pairs):
        input_tensor, target_tensor = pair_to_tensor(input_lang, output_lang, pair, device)
        input_idxes.append(input_tensor)
        target_idxes.append(target_tensor)

    train_data = TensorDataset(torch.cat(input_idxes, dim=0), torch.cat(target_idxes, dim=0))

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    return input_lang, output_lang, train_dataloader
