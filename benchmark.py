import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model.rnn import RNNNative, RNNTorch
from model.mlp import MLP, MLP2

import time

# classifier = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)

def bench_rnn_forward(batch_size=512, num_batch=10, vocab_size=1024, length=30, embed_size=128,\
                        hidden_size=128, delta=5):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rnn = RNNNative(vocab_size, embed_size, hidden_size).to(device)
    delta = np.random.randint(-delta, delta, size=num_batch)
    input = torch.LongTensor(batch_size).random_(0, vocab_size).to(device)
    hx = torch.randn(batch_size, hidden_size).to(device)
    start = time.time()
    for i in range(num_batch):
        for j in range(length + delta[i]):
            output, hx = rnn(input, hx)
    end = time.time()
    print("Elapsed time for RNNNative {:.3f}, avg length: {:.3f}".format(end - start, length + np.mean(delta)))

    rnn = RNNTorch(vocab_size, embed_size, hidden_size).to(device)
    input = torch.LongTensor(length, batch_size).random_(0, vocab_size).to(device)
    start = time.time()
    for i in range(num_batch):
        output, _ = rnn(input)
    end = time.time()
    print("Elapsed time for RNNTorch {:.3f}, avg length: {:.3f}".format(end - start, length + np.mean(delta)))

    input = torch.LongTensor(1, batch_size).random_(0, vocab_size).to(device)
    start = time.time()
    hx = torch.randn(1, batch_size, hidden_size).to(device)
    for i in range(num_batch):
        for j in range(length + delta[i]):
            output, hx = rnn(input, hx)
    end = time.time()
    print("Elapsed time for RNNTorch with variable length {:.3f}, avg length: {:.3f}".format(end - start, length + np.mean(delta)))
    
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


def bench_rnn_sample(batch_size=512, num_batch=10, vocab_size=1024, length=30, embed_size=128,\
                        hidden_size=128, delta=5):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rnn = RNNNative(vocab_size, embed_size, hidden_size).to(device)
    delta = np.random.randint(-delta, delta, size=num_batch)
    input = torch.LongTensor(batch_size).random_(0, vocab_size).to(device)
    hx = torch.randn(batch_size, hidden_size).to(device)

    start = time.time()
    with torch.no_grad():
        for i in range(num_batch):
            for j in range(length + delta[i]):
                output, hx = rnn(input, hx)
                prob_dist = torch.distributions.Categorical(logits=output) # probs should be of size batch x classes
                # import pdb; pdb.set_trace()
                input = prob_dist.sample()
    end = time.time()
    print("Elapsed time for Sampling RNNNative {:.3f}, avg length: {:.3f}".format(end - start, length + np.mean(delta)))

    hx = torch.randn(1, batch_size, hidden_size).to(device)
    input = torch.LongTensor(1, batch_size).random_(0, vocab_size).to(device)
    rnn = RNNTorch(vocab_size, embed_size, hidden_size).to(device)
    start = time.time()
    with torch.no_grad():
        for i in range(num_batch):
            for j in range(length + delta[i]):
                output, hx = rnn(input, hx)
                prob_dist = torch.distributions.Categorical(logits=output) # probs should be of size batch x classes
                # import pdb; pdb.set_trace()
                input = prob_dist.sample()
    end = time.time()
    print("Elapsed time for Sampling RNNTorch {:.3f}, avg length: {:.3f}".format(end - start, length + np.mean(delta)))


def bench_rnn_backward(batch_size=512, num_batch=100, vocab_size=1024, length=30, embed_size=128,\
                        hidden_size=128, delta=5):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rnn = RNNNative(vocab_size, embed_size, hidden_size).to(device)
    delta = np.random.randint(-delta, delta, size=num_batch)
    input = torch.LongTensor(batch_size).random_(0, vocab_size).to(device)
    label = torch.LongTensor(batch_size).random_(0, vocab_size).to(device)
    criteria = nn.CrossEntropyLoss()
    optimizer = optim.SGD(rnn.parameters(), lr=0.01)
    
    time_elapsed = 0
    for i in range(num_batch):
        input = torch.LongTensor(batch_size).random_(0, vocab_size).to(device)
        hx = torch.randn(batch_size, hidden_size).to(device)
        loss = 0
        optimizer.zero_grad()
        start = time.time()
        for j in range(length + delta[i]):
            output, hx = rnn(input, hx)
            loss += criteria(output, label)
        loss.backward()
        optimizer.step()
        time_elapsed += time.time() - start
    print("Elapsed time for RNNNative backward {:.3f}, avg length: {:.3f}".format(time_elapsed, length + np.mean(delta)))

    rnn = RNNTorch(vocab_size, embed_size, hidden_size).to(device)
    input = torch.LongTensor(length, batch_size).random_(0, vocab_size).to(device)
    label = torch.LongTensor(length*batch_size).random_(0, vocab_size).to(device)
    start = time.time()
    for i in range(num_batch):
        loss = 0
        optimizer.zero_grad()
        output, _ = rnn(input)
        loss = criteria(output.view(length*batch_size, -1), label)
        loss.backward()
        optimizer.step()
    end = time.time()
    print("Elapsed time for RNNTorch backward {:.3f}, avg length: {:.3f}".format(end - start, length + np.mean(delta)))

    
    input = torch.LongTensor(1, batch_size).random_(0, vocab_size).to(device)
    label = torch.LongTensor(batch_size).random_(0, vocab_size).to(device)
    hx = torch.randn(1, batch_size, hidden_size).to(device)
    time_elapsed = 0
    for i in range(num_batch):
        loss = 0
        input = torch.LongTensor(1, batch_size).random_(0, vocab_size).to(device)
        optimizer.zero_grad()
        start = time.time()
        for j in range(length + delta[i]):
            output, hx = rnn(input, hx)
            loss += criteria(output.view(batch_size, -1), label)
        loss.backward()
        optimizer.step()
        time_elapsed += time.time() - start
    print("Elapsed time for RNNTorch step backward {:.3f}, avg length: {:.3f}".format(time_elapsed, length + np.mean(delta)))

    # mlp 
    label = torch.LongTensor(batch_size).random_(0, vocab_size).to(device)
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

    # mlp2
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
    bench_rnn_sample(num_batch=1)
    bench_rnn_forward(num_batch=1)
    bench_rnn_backward(num_batch=1)