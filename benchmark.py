import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model.rnn import RNN
from model.mlp import MLP, MLP2

import time

# classifier = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)

def bench_rnn_forward(batch_size=64, num_batch=10, vocab_size=1024, length=30, embed_size=128, hidden_size=128):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rnn = RNN(vocab_size, embed_size, hidden_size).to(device)
    input = torch.LongTensor(batch_size).random_(0, vocab_size).to(device)
    start = time.time()
    for i in range(num_batch):
        hx = torch.randn(batch_size, hidden_size).to(device)
        for j in range(length):
            output, hx = rnn(input, hx)
    end = time.time()
    print("Elapsed time for RNN {:.3f}".format(end - start))
    mlp = MLP(device, vocab_size, embed_size, hidden_size, length).to(device)
    input = []
    hx = []
    for i in range(length):
        input.append(torch.LongTensor(batch_size).random_(0, vocab_size).to(device))
        hx.append(torch.randn(batch_size, hidden_size).to(device))
    start = time.time()
    for i in range(num_batch):
        output = mlp(input, hx)
    end = time.time()
    print("Elapsed time for MLP {:.3f}".format(end - start))



def bench_rnn_backward(batch_size=64, num_batch=100, vocab_size=1024, length=30, embed_size=128, hidden_size=128):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rnn = RNN(vocab_size, embed_size, hidden_size).to(device)
    input = torch.LongTensor(batch_size).random_(0, vocab_size).to(device)
    label = torch.LongTensor(batch_size).random_(0, vocab_size).to(device)
    criteria = nn.CrossEntropyLoss()
    optimizer = optim.SGD(rnn.parameters(), lr=0.01)
    

    start = time.time()
    for i in range(num_batch):
        loss = 0
        hx = torch.randn(batch_size, hidden_size).to(device)
        optimizer.zero_grad()
        for j in range(length):
            output, hx = rnn(input, hx)
            # import pdb; pdb.set_trace()
            loss += criteria(output, label)
        loss.backward()
        optimizer.step()
    end = time.time()
    print("Elapsed time for RNN backward {:.3f}".format(end - start))
    mlp = MLP(device, vocab_size, embed_size, hidden_size, length).to(device)
    input = []
    hx = []
    for i in range(length):
        input.append(torch.LongTensor(batch_size).random_(0, vocab_size).to(device))
        hx.append(torch.randn(batch_size, hidden_size).to(device))
    optimizer = optim.SGD(mlp.parameters(), lr=0.01)
    
    start = time.time()
    for i in range(num_batch):
        optimizer.zero_grad()
        output = mlp(input, hx)
        loss = 0
        for i in range(length):
            loss += criteria(output[i], label)
        loss.backward
        optimizer.step()
    end = time.time()
    print("Elapsed time for MLP backward {:.3f}".format(end - start))

    mlp2 = MLP2(device, vocab_size, embed_size, hidden_size, length).to(device)
    input = torch.LongTensor(batch_size).random_(0, vocab_size).to(device)
    hx = torch.randn(batch_size, hidden_size).to(device)
    optimizer = optim.SGD(mlp2.parameters(), lr=0.01)
    start = time.time()
    for i in range(num_batch):
        optimizer.zero_grad()
        loss = 0
        for i in range(length):
            output = mlp2(input, hx)
            loss += criteria(output, label)
            loss.backward
            optimizer.step()
    end = time.time()
    print("Elapsed time for MLP2 backward {:.3f}".format(end - start))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    # bench_rnn_forward(num_batch=1000)
    bench_rnn_backward(num_batch=20)