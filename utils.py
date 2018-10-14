import torch


def div_to_batch(data, batch_size):
    num_batches = data.size(0) // batch_size
    data = data.narrow(0, 0, num_batches * batch_size)
    data = data.view(batch_size, -1).t().contiguous()
    return data


def get_batch(data_source, seq_len, i):
    seq_len = min(seq_len, len(data_source) - 1 - i)
    data = data_source[i:i + seq_len]
    target = data_source[i + 1:i + 1 + seq_len].view(-1)
    return data, target


def repackage_hidden(hidden):
    if isinstance(hidden, torch.Tensor):
        return hidden.detach()
    return tuple(repackage_hidden(v) for v in hidden)


def tweet(corpus, num_tweets, device):
    temperature = 1.0
    with open('model.pt', 'rb') as f:
        model = torch.load(f).to(device)
    model.eval()

    hidden = model.init_hidden(1)
    sos_idx = corpus.dictionary.word2idx[corpus.SOS_TOKEN]
    init_input = torch.tensor([[sos_idx]], dtype=torch.long).to(device)

    with open('output.txt', 'w') as outf:
        with torch.no_grad():  # no tracking history
            count = 0
            while count < num_tweets:
                output, hidden = model(init_input, hidden)
                word_weights = output.squeeze().div(temperature).exp().cpu()
                word_idx = torch.multinomial(word_weights, 1)[0]
                input.fill_(word_idx)
                word = corpus.dictionary.idx2word[word_idx]
                outf.write(word + ' ')
                if word == corpus.EOS_TOKEN:
                    outf.write('\n')
                    count += 1

    print('Generated {} tweets'.format(num_tweets))
