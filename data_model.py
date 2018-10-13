import torch


class Dictionary():
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus():
    EOS_TOKEN = '<EOS>'

    def __init__(self, train_set, val_set, test_set):
        self.dictionary = Dictionary()
        self.train_set = self.tokenize(train_set)
        self.valid_set = self.tokenize(val_set)
        self.test_set = self.tokenize(test_set)

    def tokenize(self, dataset):
        token_idx = 0
        for line in dataset:
            tokens = line.split() + [Corpus.EOS_TOKEN]
            token_idx += len(tokens)
            for token in tokens:
                self.dictionary.add_word(token)

        ids = torch.LongTensor(token_idx)
        token_idx = 0
        for line in dataset:
            tokens = line.split() + [Corpus.EOS_TOKEN]
            for token in tokens:
                ids[token_idx] = self.dictionary.word2idx[token]
                token_idx += 1
        return ids
