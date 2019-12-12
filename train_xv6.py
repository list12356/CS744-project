import torch
import torch.nn as nn
import torch.optim as optim
import time

from torch.utils.data import DataLoader
from utils.dataset import XV6Dataset
from model.rnn import LSTMxv6

def train(seqlen=100, batch_size=128, num_epoch=1000, save_path='./xv6_ckpt.pth',\
            resume=None, lr=0.01):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = XV6Dataset(seqlen=seqlen)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    rnn = LSTMxv6().to(device)
    optimizer = optim.SGD(rnn.parameters(), lr=lr)
    # optimizer = optim.Adam(rnn.parameters(), lr=0.01)
    criteria = nn.CrossEntropyLoss()

    start_epoch = 0

    if resume != None:
        checkpoint = torch.load(resume)
        optimizer.load_state_dict(checkpoint['optim'])
        rnn.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']

    for epoch in range(start_epoch, num_epoch):
        running_loss = 0
        for sentence in dataloader:
            # import pdb; pdb.set_trace()
            actual_batch = len(sentence)
            sentence = sentence.t()
            input = sentence[0:seqlen].type(torch.LongTensor).to(device)
            label = sentence[1:seqlen + 1].type(torch.LongTensor).to(device)
            start = time.time()
            optimizer.zero_grad()
            output, _ = rnn(input)
            loss = criteria(output.view(seqlen*actual_batch, -1), label.view(seqlen*actual_batch, -1).squeeze())
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
        end = time.time()
        print("Epoch: {!s}, Loss: {:.3f},  {:.3f},".format(epoch, running_loss, end - start))
        if epoch % 30 == 0:
            sample(rnn, device)
            checkpoint = {
                'epoch': epoch,
                'state_dict': rnn.state_dict(),
                'optim': optimizer.state_dict()
            }
            torch.save(checkpoint, save_path + str(epoch))
            
def sample(rnn, device, seqlen=1000):
    input = torch.LongTensor(1, 1).random_(0, 129).to(device)
    generated = []
    with torch.no_grad():
        for i in range(seqlen):
            output, _ = rnn(input)
            prob_dist = torch.distributions.Categorical(logits=output) # probs should be of size batch x classes
            # import pdb; pdb.set_trace()
            input = prob_dist.sample()
            if input == 128:
                generated.append(10)
            else:
                generated.append(input)
    generated = bytes(generated).decode('ascii', 'ignore')
    with open('./sample.txt', 'w+') as out_f:
        out_f.write(generated)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume')
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()
    train(resume=args.resume, lr=args.lr)
