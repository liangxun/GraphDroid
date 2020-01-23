import torch
import torch.nn as nn
from torch.nn import init, Linear
from torch.autograd import Variable
import numpy as np
from sklearn import metrics
import time
import random
import os

from layers.MessagePassing import SageLayer
from layers.sampler import Sampler
from layers.attention import AttentionLayer
from setting import logger
from utils.input_data import InputData
from model_zoo.hander import BaseHander
from model_zoo.classifer import Classifer1 as Classifer
from model_zoo.sage import GraphSage


def re_view(tensor):
    """
    [batchsize, dim] --> [batchsize, 1, dim]
    """
    return tensor.view(tensor.shape[0], 1, tensor.shape[1])


class HANSage(nn.Module):
    def __init__(self, adj_lists, feat_data, num_classes, embed_dim, num_sample, num_layers, is_cuda=True):
        super(HANSage, self).__init__()
        self.is_cuda = is_cuda
        self.num_sample = num_sample
        self.embed_dim = embed_dim

        self.encoders = []
        for adj_list in adj_lists:
            self.encoders.append(GraphSage(adj_list, feat_data, num_classes, self.embed_dim, num_sample ,num_layers, self.is_cuda, as_view=True))
        for i, meta_encoder in enumerate(self.encoders):
            self.add_module('metaencoder_{}'.format(i), meta_encoder)
        
        self.atten = AttentionLayer(self.embed_dim, self.embed_dim)
        self.clf = Classifer(self.embed_dim, num_classes)
    
    def get_embedding(self, nodes):
        hiddens = [re_view(encoder.get_embedding(nodes)) for encoder in self.encoders]
        multi_embed = torch.cat(hiddens, dim=1)
        fuse_embed = self.atten(multi_embed)
        return fuse_embed
    
    def forward(self, nodes):
        fuse_embed = self.get_embedding(nodes)
        out = self.clf(fuse_embed)
        return out

    def predict(self, node, node_feat=None):
        """
        predict for single example(node) and allow feat modify of the node
        """
        hiddens = [re_view(encoder.predict(node, node_feat)) for encoder in self.encoders]
        multi_embed = torch.cat(hiddens, dim=1)
        fuse_embed = self.atten(multi_embed)
        out = self.clf(fuse_embed)
        return out.unsqueeze(0)


class HANSageHander(BaseHander):
    def __init__(self, num_class, data, args):
        self.num_class = num_class
        self.labels = data['labels']
        self.adj_lists = data['adj_lists']
        self.feat_data = data['feat_data']
        self.num_nodes, self.feat_dim = self.feat_data.shape
        self.is_cuda = args.cuda
        self.view = args.view
        self.num_sample = args.num_sample
        self.embed_dim = args.embed_dim
        self.freeze = args.freeze
        self.inputdata = InputData(self.num_nodes, self.labels, self.adj_lists, args.label_rate, self.is_cuda)
        self.train_data_loader = self.inputdata.get_train_data_load(batch_size=args.batch_size, shuffle=True)
    
    def build_model(self):
        logger.info("define model.")
        self.num_layers = 2
        self.model = HANSage(self.adj_lists, self.feat_data, self.num_class, self.embed_dim, self.num_sample, self.num_layers, self.is_cuda)
        logger.info('\n{}'.format(self.model))
        if self.is_cuda:
            self.model.cuda()

        self.custom_init(self.freeze)
        self.optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad, self.model.parameters()), lr=1e-3, weight_decay=1e-5)

        self.loss_func = nn.CrossEntropyLoss()
    
    def custom_init(self, freeze=False):
        logger.info("custom initialization. freeze={}".format(freeze))
        from setting import model_path
        import glob

        all_views = ['app_permission_app', 'app_url_app', 'app_component_app', 'app_tpl_app']

        for i in range(len(all_views)):
            checkpoint = torch.load(glob.glob(os.path.join(model_path, "*{}*".format(all_views[i])))[0])
            state_dict = checkpoint['state_dict']
            self.model.encoders[i].load_state_dict(state_dict, strict=False)
            if freeze:
                for param in self.model.encoders[i].parameters():
                    param.requires_grad = False
