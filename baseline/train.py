import sys
import os
import torch
from torch import optim
import torch.nn as nn
import numpy as np
from sklearn import metrics
import datetime
from utils import load_data_for_optim, evl_index
from mlp import bpnn
from rfn import RFN
from peernet import PeerNet
from config import *

is_cuda = torch.cuda.is_available()
if is_cuda:
    torch.cuda.set_device(3)


def train_and_test():
    # ===== set up
    # model = bpnn(feature_dim, layers_out, drop_rt)
    # model = RFN(feature_dim, layers_out, drop_rt)
    model = PeerNet(feature_dim, layers_out, drop_rt, is_cuda)
    if is_cuda:
        model.cuda()
    train_data_loader, test_data_loader = load_data_for_optim(os.path.join(data_dir, 'baseline_dataset.pkl'), batchsize)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # ===== train
    hist_loss = []
    hist_acc = []
    iter = 0
    for ep in range(epoch):
        for x, y in train_data_loader:
            model.train()
            if is_cuda:
                x = x.cuda()
                y = y.cuda()
            out = model(x)
            loss = loss_func(out, y)
            out = model(x)
            loss = loss_func(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            f1, acc, precision, recall = evl_index(y.cpu().data.numpy(), out.cpu().data.numpy().argmax(axis=1))
            hist_acc.append(acc)
            hist_loss.append(loss.item())
            print("epoch{}[iter{}]:\tloss={:.4f}\tacc={:.4f}\tf1={:.4f}".format(ep, iter, loss.item(), acc, f1))
            iter += 1
            

    # ===== test
    model.eval()
    true_y = []
    pred_y = []
    for x, y in test_data_loader:
        if is_cuda:
            x = x.cuda()
        out = model(x)
        true_y.extend(list(y.cpu().data.numpy()))
        pred_y.extend(list(out.cpu().data.numpy().argmax(axis=1)))
    f1, acc, precision, recall = evl_index(np.array(true_y), np.array(pred_y), detail=True)
    print("evaluate on test datset. \tacc={:.4f}\tf1={:.4f}\trecall={:.4f}\tprecision={:.4f}".format(acc, f1, recall, precision))


    # ===== save
    checkpoint = dict()
    checkpoint['evl_index'] = (f1, acc, precision, recall)
    checkpoint['state_dict'] = model.state_dict()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
    torch.save(checkpoint, os.path.join(model_save_dir, '{}_f1{:.4f}_{}.pt'.format(model.__class__.__name__, f1, timestamp)))
    print("finished")


if __name__ == "__main__":
    train_and_test()
