import math
import time
import torch
import utils


class Trainer:
    def __init__(self, model, train_data, val_data, test_data,
                 optimizer, criterion, vocab_size, args):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_tokens = vocab_size
        self.epochs = args.epochs
        self.clips = args.clips
        self.batch_size = args.batch_size
        self.seq_len = args.seq_len

    def train(self):
        min_loss = float('inf')

        for epoch in range(1, self.epochs+1):
            start = time.time()
            self.model.train()  # set training mode
            train_loss = 0
            hidden = self.model.init_hidden(self.batch_size)

            for batch, i in enumerate(range(0, self.train_data.size(0) - 1,
                                            self.seq_len)):
                print('epoch {}, batch {}'.format(epoch, i))
                data, targets = utils.get_batch(self.train_data,
                                                self.seq_len, i)
                hidden = utils.repackage_hidden(hidden)
                self.model.zero_grad()
                out, hidden = self.model(data, hidden)
                loss = self.criterion(out.view(-1, self.num_tokens), targets)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.clips)
                self.optimizer.step()
                train_loss += loss.item()

            val_loss = self.get_loss(self.val_data)
            print('-' * 89)
            print('Epoch {:3d} | time: {:5.2f}s | train loss {:5.2f} | '
                  'valid loss {:5.2f} | valid ppl {:8.2f}'.format(
                      epoch, (time.time() - start),
                      train_loss, val_loss, math.exp(val_loss))
                  )

            if val_loss < min_loss:
                with open('model.pt', 'wb') as f:
                    torch.save(self.model, f)
                min_loss = val_loss

    def get_loss(self, data_source):
        self.model.eval()
        total_loss = 0.
        hidden = self.model.init_hidden(self.batch_size)

        with torch.no_grad():
            for i in range(0, data_source.size(0) - 1, self.seq_len):
                data, targets = utils.get_batch(data_source, self.seq_len, i)
                output, hidden = self.model(data, hidden)
                output_flat = output.view(-1, self.num_tokens)
                total_loss += len(data) * self.criterion(output_flat,
                                                         targets).item()
                hidden = utils.repackage_hidden(hidden)
        return total_loss / len(data_source)

    def test(self):
        loss = self.get_loss(self.test_data)
        print('=' * 89)
        print('Test loss {:5.2f} | test ppl {:8.2f}'.format(loss,
                                                            math.exp(loss)))
        print('=' * 89)
