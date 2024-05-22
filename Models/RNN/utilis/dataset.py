from torch.utils.data import Dataset
from Models.RNN.utilis.preprocess import *


class IMDBDataset(Dataset):
    def __init__(self, root, train=True):
        # preprocess data if not exists
        if not os.path.exists(os.path.join(root, 'preprocessed')):
            preprocess(root, train=train)

        if train:
            data_path = os.path.join(root, 'preprocessed/train.txt')
        else:
            data_path = os.path.join(root, 'preprocessed/test.txt')

        # get preprocessed vocab
        vocab_path = os.path.join(root, 'preprocessed/vocab.pkl')
        with open(vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)
        # get data
        with open(data_path, 'r', encoding='utf-8') as f:
            data = [line.strip() for line in f.readlines()]
            self.labels = [int(line[0]) for line in data]
            self.text = [line[2:] for line in data]

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        return self.text[index], self.labels[index]
