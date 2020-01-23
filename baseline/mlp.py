import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


class bpnn(nn.Module):
    def __init__(self, feat_dim, layers_out, drop_rt):
        super(bpnn, self).__init__()
        self.drop_rt = drop_rt

        self.hidden1 = nn.Linear(feat_dim, layers_out[0])
        self.hidden2 = nn.Linear(layers_out[0], layers_out[1])
        self.out = nn.Linear(layers_out[1], layers_out[2])

    def forward(self, input):
        h1 = F.relu(self.hidden1(input))
        h1 = F.dropout(h1, self.drop_rt)
        h2 = F.relu(self.hidden2(h1))
        h2 = F.dropout(h2, self.drop_rt)
        out = self.out(h2)
        return out