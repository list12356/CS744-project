from torch import nn

class MLP(nn.Module):
    def __init__(self, device, word_size=1024, embed_size=128, hidden_size=128, seqlen=30):
        super(MLP, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.seqlen = seqlen

        self.embedding = nn.Embedding(word_size, embed_size)
        self.layers = []
        for i in range(seqlen):
            layer = {}
            layer['i2h'] = nn.Linear(embed_size, hidden_size).to(device)
            layer['h2h'] = nn.Linear(hidden_size, hidden_size).to(device)
            layer['tanh'] = nn.Tanh().to(device)
            layer['out'] = nn.Linear(hidden_size, word_size).to(device)
            self.layers.append(layer)

    def forward(self, input, hidden):
        output = []
        for i in range(self.seqlen):
            _input = self.embedding(input[i])
            i2h = self.layers[i]['i2h']
            h2h = self.layers[i]['h2h']
            tanh = self.layers[i]['tanh']
            out = self.layers[i]['out']
            h = tanh(i2h(_input) + h2h(hidden[i]))
            output.append(out(h))
        return output

class MLP2(nn.Module):
    def __init__(self, device, word_size=1024, embed_size=128, hidden_size=128, seqlen=30):
        super(MLP2, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.seqlen = seqlen

        self.embedding = nn.Embedding(word_size, embed_size)
        self.i2h = nn.Linear(embed_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()
        self.out = nn.Linear(hidden_size, word_size)
        layer = []
        for i in range(seqlen//2):
            layer.append(nn.Linear(embed_size, hidden_size).to(device))
        self.layer = nn.Sequential(*layer).to(device)

    def forward(self, input, hidden):
        input = self.embedding(input)
        hidden = self.i2h(input) + self.h2h(hidden)
        hidden = self.tanh(self.layer(hidden))
        output = self.out(hidden)
        return output