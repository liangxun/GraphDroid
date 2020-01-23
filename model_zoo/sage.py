import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init, Linear
from torch.autograd import Variable
import numpy as np
from sklearn import metrics
import time
import random
import pickle as pkl

from layers.MessagePassing import SageLayer
from layers.sampler import Sampler
from setting import logger
from utils.input_data import InputData
from model_zoo.hander import BaseHander
from model_zoo.classifer import Classifer2 as Classifer


class GraphSage(nn.Module):
    def __init__(self, adj_lists, feats, num_classes, embed_dim, num_sample, num_layers, is_cuda, as_view=False):
        super(GraphSage, self).__init__()
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.is_cuda = is_cuda
        self.as_view = as_view
        self.raw_features = torch.Tensor(feats)
        self.num_sample = num_sample
        self.embed_dim = embed_dim
        self.feat_dim = self.raw_features.shape[1]

        self.sampler = Sampler(adj_lists[0]) if type(adj_lists) == list else Sampler(adj_lists)

        self.sage_layer1 = SageLayer(in_dim=self.feat_dim, out_dim=self.embed_dim, cuda=self.is_cuda)
        self.sampler.add_sample_layer(num_sample=self.num_sample)

        self.sage_layer2 = SageLayer(in_dim=self.embed_dim, out_dim=self.embed_dim, cuda=self.is_cuda)
        self.sampler.add_sample_layer(num_sample=self.num_sample)

        if not self.as_view:
            self.clf = Classifer(self.embed_dim, self.num_classes)
    
    def get_embedding(self, nodes):
        
        nodes_layers = self.sampler.sample(nodes)

        unique_nodes_list = nodes_layers[0][0]
        pre_hidden_embs = self.raw_features[unique_nodes_list]
        if self.is_cuda:
            pre_hidden_embs = pre_hidden_embs.cuda()

        for i in range(1, self.num_layers+1):
            node = nodes_layers[i][0]
            info_neighs = nodes_layers[i-1]
            sage_layer = getattr(self, 'sage_layer{}'.format(i))
            cur_hidden_embs = sage_layer(node, pre_hidden_embs, info_neighs, i)
            pre_hidden_embs = cur_hidden_embs
        return cur_hidden_embs
    
    def forward(self, nodes):
        cur_hidden_embs = self.get_embedding(nodes)
        out = self.clf(cur_hidden_embs)
        return out

    def predict(self, node, node_feat=None):
        """
        predict for single example(node) and allow feat modify of the node
        """
        nodes_layers = self.sampler.sample(node)

        unique_nodes_list, _, unique_nodes = nodes_layers[0] 
        pre_hidden_embs = self.raw_features[unique_nodes_list].clone()
        
        if node_feat is not None:
            assert len(node) == 1
            id = node[0]
            pre_hidden_embs[unique_nodes[id]] = node_feat

        if self.is_cuda:
            pre_hidden_embs = pre_hidden_embs.cuda()

        for i in range(1, self.num_layers+1):
            node = nodes_layers[i][0]
            info_neighs = nodes_layers[i-1]
            sage_layer = getattr(self, 'sage_layer{}'.format(i))
            cur_hidden_embs = sage_layer(node, pre_hidden_embs, info_neighs, i)
            pre_hidden_embs = cur_hidden_embs

        return pre_hidden_embs if self.as_view else self.clf(cur_hidden_embs)


class GraphSageHandler(BaseHander):
    """
    wrapper for GraphSage model
    """
    def __init__(self, num_class, data, args):
        super(GraphSageHandler, self).__init__(args)
        self.num_class = num_class
        self.labels = data['labels']
        self.adj_lists = data['adj_lists']
        self.feat_data = data['feat_data']
        self.num_nodes, self.feat_dim = self.feat_data.shape
        self.is_cuda = args.cuda
        self.view = args.view
        self.num_sample = args.num_sample
        self.embed_dim = args.embed_dim
        self.dropout = args.dropout
        self.inputdata = InputData(self.num_nodes, self.labels, self.adj_lists, args.label_rate, self.is_cuda)
        self.train_data_loader = self.inputdata.get_train_data_load(batch_size=args.batch_size, shuffle=True)
    
    def build_model(self):
        logger.info("define model.")
        self.num_layers = 2
        self.model = GraphSage(self.adj_lists, self.feat_data, self.num_class, self.embed_dim, self.num_sample, self.num_layers, self.is_cuda)
        logger.info('\n{}'.format(self.model))
        if self.is_cuda:
            self.model.cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-5)
        self.loss_func = nn.CrossEntropyLoss()
