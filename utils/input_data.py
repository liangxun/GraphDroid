import os
import numpy as np
import random
from setting import logger, data_dir
import sys
import torch
import json
from torch.utils.data import TensorDataset, DataLoader


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


class InputData(object):
    def __init__(self, num_nodes, labels, adj_lists, label_rate, cuda):
        self.num_nodes = num_nodes
        self.adj_lists = adj_lists
        self.labels = labels
        self.cuda = cuda
        self.split_data(label_rate)
    
    def get_train_data_load(self, batch_size, shuffle=True):
        x_ids = self.train_labeled
        y = self.labels[x_ids]
        x_ids = torch.LongTensor(x_ids).unsqueeze(1)
        y = torch.LongTensor(y).unsqueeze(1)
        if self.cuda:
            x_ids = x_ids.cuda()
            y = y.cuda()
        dataset = TensorDataset(x_ids, y)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return data_loader

    def gen_train_batch(self, batch_size):
        ind = self.train_labeled
        while True:
            random.shuffle(ind)
            i = 0
            while i < len(ind):
                j = min(len(ind), i + batch_size)
                x = ind[i:j]
                y = self.labels[x]
                yield x, y
                i = j
    
    def get_val_data(self):
        x = self.val
        y = self.labels[x]
        return x, y
    
    def get_test_data(self):
        x = self.test
        y = self.labels[x]
        return x, y 


    def split_data(self, label_rate=1):
        with open(os.path.join(data_dir, 'label_info', 'split_info.json'), 'r') as f:
            split_info = json.load(f)
        ids_train = split_info['train_ids']
        ids_val = split_info['val_ids']
        ids_test = split_info['test_ids']
    
        self.test = ids_test
        self.train_all = ids_train
        self.train_labeled = ids_train[: round(len(ids_train)*label_rate)]
        self.val = ids_val
        self.test = ids_test
        
        self.mask_train_labeled = sample_mask(self.train_labeled, self.num_nodes)
        logger.info("label_rate={}".format(label_rate))
        logger.info("(train : val : test) = ({} : {} : {})".format(len(self.train_all), len(self.val), len(self.test)))
        logger.info("{} labeded samples in train set.".format(len(self.train_labeled)))
