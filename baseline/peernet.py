import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsnooper


class PeerNet(nn.Module):
    def __init__(self, input_dim, layers_out, drop_rate=0.4, is_cuda=True):
        super(PeerNet, self).__init__()
        self.drop_rt = drop_rate
        self.is_cuda = is_cuda

        self.hidden1 = nn.Linear(input_dim, layers_out[0])
        self.pr_layer = PRLayer(layers_out[0], layers_out[0], num_neigh=5+1, cuda=self.is_cuda)
        self.hidden2 = nn.Linear(layers_out[0], layers_out[1])
        self.out = nn.Linear(layers_out[1], layers_out[2])
    
    def forward(self, input):
        h1 = self.hidden1(input)
        h1 = F.relu(h1)
        h1 = F.dropout(h1, self.drop_rt)

        pr_h = self.pr_layer(h1)
        pr_h = F.relu(pr_h)

        h2 = self.hidden2(pr_h)
        h2 = F.relu(h2)
        h2 = F.dropout(h2, self.drop_rt)
        out = self.out(h2)
        return out


class PRLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_neigh, cuda=True):
        super(PRLayer, self).__init__()
        self.cuda = cuda
        self.linear = nn.Linear(in_dim, out_dim)
        self.num_neigh = num_neigh
    
    #@torchsnooper.snoop()
    def transfom(self, inputs):
        out = torch.zeros_like(inputs)
        batch_size = inputs.shape[0]

        for i in range(inputs.shape[1]):
            line = inputs[:, i]
            assert line.shape[0] == batch_size

            tmp = line.view(batch_size, 1).repeat(1,batch_size)
            indexs = abs(line - tmp).argsort(1)[:, :self.num_neigh]
            row = [i for i in range(indexs.shape[0]) for _ in range(indexs.shape[1])]
            col = [i.item() for j in indexs for i in j]

            mask = torch.zeros(batch_size, batch_size)
            mask[row, col] = 1
            mask = mask.div(self.num_neigh)
            if self.cuda:
                mask = mask.cuda()
            out[:, i] = mask.mm(line.view(batch_size,1)).squeeze()
        return out            

    def forward(self, inputs):
        pr_input = self.transfom(inputs)
        out = self.linear(pr_input)
        return out
