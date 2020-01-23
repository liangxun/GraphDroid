"""
Adversarial Training
"""
import sys
import os
import random
import torch
from torch import optim
import torch.nn as nn
import numpy as np
from sklearn import metrics
import datetime
from utils import load_data_for_optim, evl_index
from mlp import bpnn
from rfn import RFN
from adv.batchadv import gen_adv_for_batch
from config import *

is_cuda = torch.cuda.is_available()
if is_cuda:
    torch.cuda.set_device(cuda_device_id)

def init_params(model, pre_train_model_file):
    checkpoint = torch.load(pre_train_model_file)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    return model


def train_and_test(epoch, pre_train_model_file, alg, max_bit):
    # ===== set up
    model = bpnn(feature_dim, layers_out, drop_rt)
    model = init_params(model, pre_train_model_file)
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
            modify_bit = random.sample(list(range(10, max_bit+10, 10)), 1)[0]
            x, y = gen_adv_for_batch(model, x, y, alg, modify_bit, is_cuda)
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
            #break

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
    torch.save(checkpoint, os.path.join(model_save_dir, 'retrain_{}_f1{:.4f}_{}.pt'.format(model.__class__.__name__, f1, timestamp)))
    print("finished")


if __name__ == "__main__":
    pretrain_model_file = os.path.join(model_save_dir, 'bpnn_f10.9936_2019-12-31.pt')
    alg = 'fgsm'
    max_bit = 40
    epoch = 2
    train_and_test(epoch, pretrain_model_file, alg, max_bit)
