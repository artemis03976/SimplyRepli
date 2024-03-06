import os
import pickle
from tqdm import tqdm

from global_utilis.normalize_string import *

min_count = 5

labels = {
    'pos': '1',
    'neg': '0'
}

UNK_token = 0
PAD_token = 1
vocab = {
    "UNK": UNK_token,
    "PAD": PAD_token,
}


def preprocess(text_dir, train=True):
    print("preprocessing {} data...".format("training" if train else "testing"))

    data_dir = os.path.join(text_dir + 'train') if train else os.path.join(text_dir + 'test')
    file_name = 'train.txt' if train else 'test.txt'

    save_dir = os.path.join(text_dir, 'preprocessed')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    save_path = os.path.join(save_dir, file_name)

    with open(save_path, 'w', encoding='utf-8') as preprocessed_file:
        word_count = {}

        for dir_name in labels.keys():
            label = labels[dir_name]
            class_dir = os.path.join(data_dir, dir_name)

            for file in tqdm(os.listdir(class_dir)):
                if not file.endswith('txt'):
                    continue

                lines = open(os.path.join(class_dir, file), 'r', encoding='utf-8').read()
                lines = normalize_string(lines)
                tokens = lines.split()  # 分词统计词数
                for token in tokens:
                    if token in word_count.keys():
                        word_count[token] = word_count[token] + 1
                    else:
                        word_count[token] = 0

                preprocessed_file.write(label + ' ' + lines + '\n')
                preprocessed_file.flush()

    if train:
        print("building vocabulary...")

        for word, count in word_count.items():
            if count >= min_count:
                if word not in vocab.keys():
                    vocab[word] = len(vocab)

        print("finish building vocabulary...")

        with open(os.path.join(save_dir, 'vocab.pkl'), 'wb') as f:
            pickle.dump(vocab, f)

        print("vocabulary saved...")


if __name__ == '__main__':
    text_dir = '../../../datas/IMDB/'
    preprocess(text_dir, train=True)
    preprocess(text_dir, train=False)
