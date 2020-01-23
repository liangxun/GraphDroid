import os
import pickle as pkl
import torch
import json
from sklearn import metrics
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from model_zoo.sage import GraphSage
from model_zoo.han_sage import HANSage
from hin.load_data import load_data
from setting import *


# ============================= load pretrained model ===========================================
def load_pretrain_model(model_file):
    """
    load pretrained model
    """
    model_type = 'GraphSage'
    model_view = os.path.split(model_file)[1].split('_')[-3]
    logger.debug(model_view)
    if model_view == 'han':
        model_type = 'HANSage'
        adj_lists, feat_data, labels = load_data(data_dir)
    else:
        adj_lists, feat_data, labels = load_data(data_dir, 'app_{}_app'.format(model_view))
    
    # logger.info("loading dataset.")
    num_class = 2
    embed_dim = 200
    num_sample = 5
    num_layers = 2
    is_cuda = torch.cuda.is_available()

    map_guessmodel = {
        'GraphSage': GraphSage,
        'HANSage': HANSage,
    }
    model_class = map_guessmodel[model_type]
    model = model_class(adj_lists, feat_data, num_class, embed_dim, num_sample, num_layers, is_cuda)
    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint['state_dict'])
    if is_cuda:
        model.cuda()
    model.eval()
    return model, torch.Tensor(feat_data)


def load_data_for_adv(batch_size=1):
    with open(os.path.join(data_dir, 'label_info', 'split_info.json'), 'r') as f:
        split_info = json.load(f)
    test_ids = split_info['test_ids']
    test_y = split_info['test_y']
    x_ids = torch.LongTensor(test_ids).unsqueeze(1)
    y = torch.LongTensor(test_y).unsqueeze(1)
    dataset = TensorDataset(x_ids, y)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)    
    return data_loader


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
