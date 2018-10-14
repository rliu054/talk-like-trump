import argparse
import glob

import torch
import torch.nn as nn

import utils
from data_model import Corpus
from data_processor import DataProcessor
from model import RNN
from trainer import Trainer

parser = argparse.ArgumentParser(description='Generate tweets in Trump style')
parser.add_argument('--data', type=str, default='./data',
                    help='data corpus location')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size')
parser.add_argument('--epochs', type=int, default=20,
                    help='number of epochs to run')
parser.add_argument('--embed_size', type=int, default=200,
                    help='embedding size')
parser.add_argument('--hidden_size', type=int, default=200,
                    help='hidden layer size')
parser.add_argument('--num_layers', type=int, default=2,
                    help='layers of RNN to be stacked')
parser.add_argument('--seq_len', type=int, default=15,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout ratio')
parser.add_argument('--clips', type=float, default=0.25,
                    help='used to clip gradients')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate')

args = parser.parse_args()

# -- use gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -- generate data set
file_list = glob.glob('./data/*.json')
raw_datasets = DataProcessor(file_list).generate_datasets()

# -- tokenize
corpus = Corpus(raw_datasets)
datasets = corpus.tokenize_all()
vocab_size = len(corpus.dictionary)

# --- tokenize data
train_data = utils.div_to_batch(datasets[0], args.batch_size).to(device)
val_data = utils.div_to_batch(datasets[1], args.batch_size).to(device)
test_data = utils.div_to_batch(datasets[2], args.batch_size).to(device)

# -- define model
model = RNN(vocab_size, args.embed_size, args.hidden_size,
            args.num_layers, args.dropout).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# -- train and validate model
trainer = Trainer(model, train_data, val_data, test_data,
                  optimizer, criterion, vocab_size, args)
trainer.train()
trainer.test()

# -- generate tweets
utils.tweet(corpus, 100, device)
