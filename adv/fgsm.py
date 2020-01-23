import torch
import torch.nn as nn
import numpy as np
from setting import logger


class FGSM:
    def __init__(self, model, max_bit=10, is_cuda=False):
        self.model = model
        self.max_bit = max_bit
        self.is_cuda = is_cuda

        self.loss_func = nn.CrossEntropyLoss()
        self.model.eval()
        if self.is_cuda:
            self.model = self.model.cuda()
    
    def generate(self, data, data_grad):
        """
        :parma k表示允许修改的bit数量
        """
        data = data.data.numpy().squeeze()
        data_grad = data_grad.numpy().squeeze()
        #mask_data = np.array(list(map(lambda i: True if i==0 else False, data)))
        mask_data = (data == 0)
        #mask_grad = np.array(list(map(lambda x: True if x>0 else False, data_grad)))
        mask_grad = (data_grad > 0)
        mask = mask_data & mask_grad
        data_grad[~mask] = 0

        index = data_grad.argsort()[-self.max_bit:]
        data[index] = 1
        data = torch.Tensor(data).unsqueeze(0)
        return data
    
    def attack(self, id, x, true_y):
        """
        return: 0表示本来模型就误判，不需要生成对抗样本。 -1表示对抗样本生成失败。 n(n>0)表示修改n个bit，成功生成对抗样本。
        """
        if self.is_cuda:
            x = x.cuda()
        x.requires_grad = True
        out = self.model.predict(id, x)
        init_pred = out.cpu().max(1, keepdim=True)[1]
        if init_pred.item() != true_y.item(): # 本来模型就判错的那些就不用管了。
            #print("no need! init_pred={}, true_y={}".format(init_pred.item(), true_y.item()))
            return 0, None
        loss = self.loss_func(out, true_y.cuda() if self.is_cuda else true_y)
        self.model.zero_grad
        loss.backward()
        x_grad = x.grad.data
        adv_x = self.generate(x.cpu(), x_grad.cpu())
        out = self.model.predict(id, adv_x)
        adv_pred = out.cpu().max(1, keepdim=True)[1]
        if adv_pred.item() != true_y.item(): # 成功生成对抗样本，返回生成的对抗样本
            #print("success!")
            return self.max_bit, adv_x.cpu().data.numpy().squeeze()
        else:
            #print("fialed!")    # 生成对抗样本失败，返回原来的feat
            return -1, None
