import os
import pickle
from tqdm import tqdm

from global_utilis.normalize_string import *

min_count = 5  # words with a frequency of less than 5 are replaced with UNK

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

    # get file path
    data_dir = os.path.join(text_dir + 'train') if train else os.path.join(text_dir + 'test')
    file_name = 'train.txt' if train else 'test.txt'

    # create save directory
    save_dir = os.path.join(text_dir, 'preprocessed')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_path = os.path.join(save_dir, file_name)

    with open(save_path, 'w', encoding='utf-8') as preprocessed_file:
        word_count = {}

        for dir_name in labels.keys():
            # browse by label as directory name
            label = labels[dir_name]
            class_dir = os.path.join(data_dir, dir_name)

            for file in tqdm(os.listdir(class_dir)):
                # skip irrelavant file
                if not file.endswith('txt'):
                    continue
                # get sentence
                lines = open(os.path.join(class_dir, file), 'r', encoding='utf-8').read()
                lines = normalize_string(lines)
                # split sentence with space to count words
                tokens = lines.split()
                for token in tokens:
                    if token in word_count.keys():
                        word_count[token] += 1
                    else:
                        # create an entry
                        word_count[token] = 0
                # write to file
                preprocessed_file.write(label + ' ' + lines + '\n')
                preprocessed_file.flush()

    # build vocab for training data
    if train:
        print("building vocabulary...")
        # filter words below certain frequency
        for word, count in word_count.items():
            if count >= min_count:
                if word not in vocab.keys():
                    vocab[word] = len(vocab)

        print("finish building vocabulary...")
        # save vocab
        with open(os.path.join(save_dir, 'vocab.pkl'), 'wb') as f:
            pickle.dump(vocab, f)

        print("vocabulary saved...")


if __name__ == '__main__':
    text_dir = '../../../datas/IMDB/'
    preprocess(text_dir, train=True)
    preprocess(text_dir, train=False)
