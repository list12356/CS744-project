import numpy as np
from torch.utils.data import Dataset


class XV6Dataset(Dataset):
    """xv6 dataset."""

    def __init__(self, eof=False, seqlen=100):
        """
        Args:
        """
        if eof == False:
            text = open('./data/xv6').read()
            text = list(text.encode('ascii'))
        if len(text) % seqlen != 0:
            text.extend([128]*(len(text) % seqlen))
        self.text = text
        self.seqlen = seqlen
        self.length = len(self.text) // seqlen
        # self.text = np.reshape(text, (len(text) // seqlen, seqlen))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        text = self.text[idx*self.seqlen: idx*self.seqlen + self.seqlen]
        if idx == self.length - 1:
            text.append(128)
        else:
            text.append(self.text[idx*self.seqlen + self.seqlen])
        return np.array(text)