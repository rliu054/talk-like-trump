import json
import re

import torch
from torch.utils.data import random_split


class DataProcessor():
    def __init__(self, file_list, dump_file='dump.txt'):
        self.file_list = file_list
        self.dump_file = dump_file

    def generate_dataset(self, lengths=(0.9, 0.05, 0.05)):
        assert sum(lengths) == 1, 'sum of lengths does not equal to 1'

        dataset, total_size = self._preprocess()
        train_size = int(lengths[0] * total_size)
        val_size = int(lengths[1] * total_size)
        test_size = total_size - train_size - val_size
        train_set, val_set, test_set = random_split(
            dataset, [train_size, test_size, val_size]
        )

        return train_set, val_set, test_set

    def _preprocess(self):
        total_size = 0
        full_dataset = []

        out_file = open(self.dump_file, 'w', encoding='utf8')
        for file in self.file_list:
            in_file = open(file, 'r', encoding='utf8')
            f_json = json.load(in_file)
            for json_obj in f_json:
                if not json_obj['is_retweet']:
                    line = re.sub('(http|www|@|#)\S+', '', json_obj['text'])
                    line = re.sub('[^\w\s]', '', line)
                    total_size += 1
                    out_file.write(line.lower() + "\n")
                    full_dataset.append(line.lower())
            in_file.close()
        out_file.close()

        return full_dataset, total_size
