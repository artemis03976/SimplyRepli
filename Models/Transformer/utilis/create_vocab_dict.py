import os
from io import open
import jieba
import pickle
from global_utilis.normalize_string import *

SOS_token = 0
EOS_token = 1
PAD_token = 2
UNK_token = 3
MAX_LENGTH = 40  # maximum length of a sentence


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "PAD"}
        self.n_words = 3  # Count SOS, EOS and PAD

    def add_sentence(self, sentence):
        # use jieba for Chinese segmentation
        if self.name == "cmn":
            words = jieba.cut(sentence)
        else:
            words = sentence.split(' ')
        for word in words:
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            # create entry
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def read_data(lang1, lang2, reverse=False):
    print("Reading lines...")

    # read the file and split into lines
    file_path = '../../datas/machine_translation/%s-%s.txt'
    lines = open(file_path % (lang1, lang2), encoding='utf-8').read().strip().split('\n')

    # split every line into pairs and normalize
    pairs = []
    for line in lines:
        pair = line.split('\t')
        pair[0] = normalize_string(pair[0])
        pair[1] = unicode_to_ascii(pair[1].lower().strip())
        pairs.append(pair)

    # reverse pairs
    pairs = [list(reversed(p)) for p in pairs] if reverse else pairs

    return pairs


def save_vocab(lang, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(lang, f)


def load_vocab(save_path):
    with open(save_path, 'rb') as f:
        return pickle.load(f)


def create_lang_vocab(lang1, lang2, pairs, reverse):
    if reverse:
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])

    return input_lang, output_lang


def prepare_data(lang1, lang2, reverse=False):
    # create save directory
    current_directory = os.path.dirname(os.path.abspath(__file__))
    save_dir = '../../../datas/machine_translation/vocab/'
    input_vocab_file = lang2 + '.pkl' if reverse else lang1 + '.pkl'
    output_vocab_file = lang1 + '.pkl' if reverse else lang2 + '.pkl'

    input_vocab_path = os.path.join(current_directory, save_dir + input_vocab_file)
    output_vocab_path = os.path.join(current_directory, save_dir + output_vocab_file)

    pairs = read_data(lang1, lang2, reverse)

    # read vocab if already exists, else create
    if os.path.exists(input_vocab_path) and os.path.exists(output_vocab_path):
        input_lang = load_vocab(input_vocab_path)
        output_lang = load_vocab(output_vocab_path)
    else:
        input_lang, output_lang = create_lang_vocab(lang1, lang2, pairs, reverse)
        save_vocab(input_lang, input_vocab_path)
        save_vocab(output_lang, output_vocab_path)

    print("Read %s sentence pairs" % len(pairs))
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)

    return input_lang, output_lang, pairs
