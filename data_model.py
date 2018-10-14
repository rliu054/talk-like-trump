import torch


class Dictionary:
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


class Corpus:
    SOS_TOKEN = '<SOS>'
    EOS_TOKEN = '<EOS>'

    def __init__(self, data_sets):
        self.dictionary = Dictionary()
        self.data_sets = data_sets

    def tokenize_all(self):
        tokenized_sets = []
        for data_set in self.data_sets:
            tokenized_sets.append(self._tokenize(data_set))
        return tokenized_sets

    def _tokenize(self, data_set):
        token_idx = 0
        for line in data_set:
            tokens = [Corpus.SOS_TOKEN] + line.split() + [Corpus.EOS_TOKEN]
            token_idx += len(tokens)
            for token in tokens:
                self.dictionary.add_word(token)

        ids = torch.LongTensor(token_idx)
        token_idx = 0
        for line in data_set:
            tokens = line.split() + [Corpus.EOS_TOKEN]
            for token in tokens:
                ids[token_idx] = self.dictionary.word2idx[token]
                token_idx += 1
        return ids
