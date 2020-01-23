import torch
import torch.nn as nn
import numpy as np


class FGSM:
    def __init__(self, model, max_bit, is_cuda=False):
        self.model = model
        self.max_bit = max_bit
        self.is_cuda = is_cuda

        self.loss_func = nn.CrossEntropyLoss()
        self.model.eval()
        if self.is_cuda:
            self.model = self.model.cuda()
    
    def generate(self, data, data_grad):
        data = data.data.numpy().squeeze()
        data_grad = data_grad.numpy().squeeze()
        mask_data = (data == 0)
        mask_grad = (data_grad > 0)
        mask = mask_data & mask_grad
        data_grad[~mask] = 0

        index = data_grad.argsort()[-self.max_bit:]
        data[index] = 1
        data = torch.Tensor(data).unsqueeze(0)
        return data
    
    def attack(self, x, true_y):
        if self.is_cuda:
            x = x.cuda()
        x.requires_grad = True
        out = self.model(x)
        init_pred = out.cpu().max(1, keepdim=True)[1]
        if init_pred.item() != true_y.item():
            return 0, x.detach().cuda() if self.is_cuda else x.detach().cpu()

        loss = self.loss_func(out, true_y.cuda()) if self.is_cuda else self.loss_func(out, true_y)
        self.model.zero_grad
        loss.backward()
        x_grad = x.grad.data
        adv_x = self.generate(x.cpu(), x_grad.cpu())
        out = self.model(adv_x.cuda()) if self.is_cuda else self.model(adv_x)
        adv_pred = out.cpu().max(1, keepdim=True)[1]
        if adv_pred.item() != true_y.item():
            return self.max_bit, adv_x.cuda() if self.is_cuda else adv_x.cpu()
        else:
            return -1, adv_x.cuda() if self.is_cuda else adv_x.cpu()
