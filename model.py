import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, vocab_size, embed_size,
                 hidden_size, num_layers, dropout):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed = nn.Embedding(vocab_size, embed_size)
        # self.lstm = nn.LSTM(embed_size, hidden_size,
        #                     num_layers, bidirectional=True)
        self.lstm = nn.LSTM(embed_size, hidden_size,
                            num_layers)
        self.dropout = nn.Dropout(dropout)
        # self.linear = nn.Linear(hidden_size * 2, vocab_size)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, hidden):
        embed = self.embed(x)
        x = self.dropout(embed)
        out, hidden = self.lstm(x, hidden)
        out = self.dropout(out)
        # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        decoded = self.linear(out.view(out.size(0)*out.size(1), out.size(2)))
        return decoded.view(out.size(0), out.size(1), decoded.size(1)), hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers,
                                 batch_size, self.hidden_size),
                weight.new_zeros(self.num_layers,
                                 batch_size, self.hidden_size))
    # def init_hidden(self, batch_size):
    #     weight = next(self.parameters())
    #     return (weight.new_zeros(self.num_layers * 2,
    #                              batch_size, self.hidden_size),
    #             weight.new_zeros(self.num_layers * 2,
    #                              batch_size, self.hidden_size))
