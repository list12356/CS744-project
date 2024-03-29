from torch import nn
import torch
import torch.nn.functional as F


class RNNNative(nn.Module):
    """
        Native pytorch implementation of Elman RNN
    """
    def __init__(self, word_size=1024, embed_size=128, hidden_size=128):
        super(RNNNative, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size

        self.embedding = nn.Embedding(word_size, embed_size)
        self.i2h = nn.Linear(embed_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()
        self.out = nn.Linear(hidden_size, word_size)

    def forward(self, input, hidden):
        input = self.embedding(input)
        hidden = self.tanh(self.i2h(input) + self.h2h(hidden))
        output = self.out(hidden)
        return output, hidden

class RNNTorch(nn.Module):
    """
        Using built in torch.nn.RNN module
    """
    def __init__(self, word_size=1024, embed_size=128, hidden_size=128):
        super(RNNTorch, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size

        self.embedding = nn.Embedding(word_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size)
        self.out = nn.Linear(hidden_size, word_size)

    def forward(self, input, hidden=None):
        """
            input: seqlen x batch
        """
        embed = self.embedding(input)
        rnn_out, hidden = self.rnn(embed, hidden)
        output = self.out(rnn_out)
        return output, hidden

class LSTMxv6(nn.Module):
    def __init__(self, word_size=129, hidden_size=512, num_layer=1):
        super(LSTMxv6, self).__init__()
        self.word_size = word_size
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(word_size, hidden_size, num_layer)
        self.out = nn.Linear(hidden_size, word_size)

    def forward(self, input, hidden=None):
        # import pdb; pdb.set_trace()
        onehot = F.one_hot(input, num_classes=self.word_size).float() # seqlen x batch x word_size
        rnn_out, hidden = self.rnn(onehot, hidden)
        output = self.out(rnn_out)
        return output, hidden