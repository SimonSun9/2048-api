import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, SubsetRandomSampler
from torch import nn
from torch.autograd import Variable
import numpy as np


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(
            input_size=4,
            hidden_size=500,
            num_layers=4,
            batch_first=True,
        )
        self.hidden2 = nn.Sequential(nn.Linear(500, 100), nn.ReLU(True))

        self.out = nn.Linear(100, 4)

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)

        out = self.hidden2(r_out[:, -1, :])
        out = self.out(out)
        return out
