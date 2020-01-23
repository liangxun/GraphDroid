import torch
import torch.nn as nn
import torch.nn.functional as F


class Classifer1(nn.Module):
    def __init__(self, in_dim, out_dim, drop_rate=0.3):
        super(Classifer1, self).__init__()

        self.fc1 = nn.Linear(in_dim, out_dim)

    def forward(self, input):
        out = self.fc1(input)
        return out


class Classifer2(nn.Module):
    def __init__(self, in_dim, out_dim, drop_rate=0.3):
        super(Classifer2, self).__init__()
        self.drop_rt = drop_rate

        self.fc1 = nn.Linear(in_dim, in_dim)
        self.out = nn.Linear(in_dim, out_dim)

    def forward(self, input):
        h1 = self.fc1(input)
        h1 = F.relu(h1)
        h1 = F.dropout(h1, self.drop_rt)
        out = self.out(h1)
        return out


class Classifer3(nn.Module):
    def __init__(self, in_dim, out_dim, drop_rate=0.3):
        super(Classifer3, self).__init__()
        self.drop_rt = drop_rate

        self.fc1 = nn.Linear(in_dim, round((out_dim+in_dim)/2))
        self.fc2 = nn.Linear(round((out_dim+in_dim)/2), round((out_dim+in_dim)/2))
        self.out = nn.Linear(round((out_dim+in_dim)/2), out_dim)

    def forward(self, input):
        h1 = self.fc1(input)
        h1 = F.relu(h1)
        h1 = F.dropout(h1, self.drop_rt)
        h2 = self.fc2(h1)
        h2 = F.relu(h2)
        h2 = F.dropout(h2, self.drop_rt)
        out = self.out(h2)
        return out
