import os
import pickle as pkl
from scipy import sparse as sp
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn import metrics
from mlp import bpnn
from rfn import RFN
from peernet import PeerNet
from config import *



# ==================metrics =========================
def evl_index(true_y, pred_y, detail=False):
    """
    f1, accuracy, precision, recall
    confusion_matrix
    """
    f1 = metrics.f1_score(true_y, pred_y)
    acc = metrics.accuracy_score(true_y, pred_y)
    precision = metrics.precision_score(true_y, pred_y)
    recall = metrics.recall_score(true_y, pred_y)
    if detail is True:
        print("\nConfusion_Matrix:\n{}".format(metrics.confusion_matrix(true_y, pred_y)))
    return f1, acc, precision, recall


def evl_index_for_adv(r_codes, alg):
    print("{} adv_attack.".format(alg))
    r_codes = np.array(r_codes)
    print(r_codes)
    rc_0 = np.sum(r_codes==0)   
    rc_n = np.sum(r_codes==-1)  
    rc_p = np.sum(r_codes>0)
    assert len(r_codes) == rc_0 + rc_n + rc_p
    print("rc_0={}\trc_n={}\trc_p={}".format(rc_0, rc_n, rc_p))
    recall_orig = (rc_n + rc_p) / (rc_0 + rc_n + rc_p)
    recall_adv = rc_n / (rc_0 + rc_n + rc_p)
    foolrate = rc_p / (rc_n + rc_p)
    print("recall_orig={:.4f}\trecall_adv={:.4f}\tfoolrate={:.4f}".format(recall_orig, recall_adv, foolrate))



# ==================load dataset============================
def transfor(input):
    """
    scr_matrix ---> numpy.narray
    """
    if type(input) == sp.csr_matrix:
        ret = np.array(input.todense(), dtype=np.float32).squeeze()
    else:
        ret = np.array(input, dtype=np.float32).squeeze()
    return ret


class MyDataset(Dataset):
    def __init__(self, x, y):
        super(MyDataset, self).__init__()
        self.x = x
        self.y = y
        assert self.x.shape[0] == len(y)
    
    def __getitem__(self, index):
        return torch.Tensor(transfor(self.x[index])), torch.LongTensor(transfor(self.y[index]))
    
    def __len__(self):
        return self.x.shape[0]


def load_data_for_optim(data_file, batch_size):
    with open(data_file, 'rb') as f:
        data = pkl.load(f)
    train_x, train_y = data['train_pairs']
    val_x, val_y = data['val_pairs']
    test_x, test_y = data['test_pairs']

    train_dataset = MyDataset(train_x, train_y)
    test_dataset = MyDataset(test_x, test_y)
    
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_data_loader, test_data_loader


def load_data_for_adv(data_file, batch_size=1): 
    with open(data_file, 'rb') as f:
        data = pkl.load(f)
    test_x, test_y = data['test_pairs']
    test_dataset = MyDataset(test_x, test_y)
    
    data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return data_loader


def load_pretrain_model(model_file):
    map_guessmodel = {
        'bpnn': bpnn,
        'RFN': RFN,
        'PeerNet': PeerNet,
    }
    model_type = os.path.split(model_file)[1].split('_')[0]
    model_class = map_guessmodel[model_type]
    model = model_class(feature_dim, layers_out, drop_rt)
    
    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model


def load_train_dataset(data_file):
    with open(data_file, 'rb') as f:
        data = pkl.load(f)
    train_x, train_y = data['train_pairs']
    return torch.Tensor(train_x.todense())

    