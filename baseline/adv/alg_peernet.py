"""
adversarial attacks for peernet
"""
import torch
import torch.nn as nn
import numpy as np
import random
import torchsnooper


def catch_batch_samples(dataset, batch_size=128):
    indexs = sorted(random.sample(range(dataset.shape[0]), batch_size))
    batch_x = dataset[indexs].clone()
    return batch_x


class FGSM:
    def __init__(self, model, train_data, max_bit, batchsize=128, is_cuda=True):
        self.model = model
        self.max_bit = max_bit
        self.is_cuda = is_cuda

        self.loss_func = nn.CrossEntropyLoss()
        self.model.eval()
        if self.is_cuda:
            self.model.cuda()
        self.set_x = torch.Tensor(train_data)
        self.batchsize = batchsize
    
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
    
    #@torchsnooper.snoop()
    def attack(self, x, true_y):
        batch_x = catch_batch_samples(self.set_x, self.batchsize)
        x.requires_grad = True
        batch_x[0] = x
        if self.is_cuda:
            batch_x = batch_x.cuda()
        batch_out = self.model(batch_x)
        out = batch_out[0].unsqueeze(0)
        init_pred = out.cpu().max(1, keepdim=True)[1]
        if init_pred.item() != true_y.item():
            return 0, x.cuda() if self.is_cuda else x.cpu()

        loss = self.loss_func(out, true_y.cuda()) if self.is_cuda else self.loss_func(out, true_y)
        self.model.zero_grad
        loss.backward()
        x_grad = x.grad.data
        adv_x = self.generate(x.cpu(), x_grad.cpu())

        batch_x[0] = adv_x
        batch_out = self.model(batch_x)
        out = batch_out[0].unsqueeze(0)

        adv_pred = out.cpu().max(1, keepdim=True)[1]
        
        if adv_pred.item() != true_y.item():
            return self.max_bit, adv_x.cuda() if self.is_cuda else adv_x.cpu()
        else:
            return -1, adv_x.cuda() if self.is_cuda else adv_x.cpu()


class JSMA:
    def __init__(self, model, train_data, max_bit, batchsize=100, is_cuda=True):
        self.max_bit = max_bit
        self.is_cuda = is_cuda
        self.model = model
        self.model.eval()
        if self.is_cuda:
            self.model.cuda()

        self.set_x = torch.Tensor(train_data)
        self.batchsize = batchsize

    def generate(self, data, data_grad, y):
        data = data.data.numpy().squeeze()
        data_grad_0 = data_grad[0].numpy().squeeze()
        data_grad_1 = data_grad[1].numpy().squeeze()
        y = y.item()
        assert data.shape[0] == data_grad_0.shape[0] == data_grad_1.shape[0]

        # compute saliency map.
        """
        <<The Limitations of Deep Learning in Adversarial Settings>> formula(8)
        """
        if y == 1:
            s_map_0 = np.zeros_like(data)
            for i in range(s_map_0.shape[0]):
                if data_grad_0[i] < 0 or data_grad_1[i] > 0:
                    pass
                else:
                    s_map_0[i] = data_grad_0[i] * abs(data_grad_1[i])
            s_map = s_map_0
        elif y == 0: 
            s_map_1 = np.zeros_like(data)
            for i in range(s_map_1.shape[0]):
                if data_grad_1[i] < 0 or data_grad_0[i] > 0:
                    pass
                else:
                    s_map_1[i] = data_grad_1[i] * abs(data_grad_0[i])
            s_map = s_map_1
        else:
            raise("label is wrong")
        
        mask_data = (data == 0)
        mask_smap = (s_map > 0)
        mask = mask_data & mask_smap
        s_map[~mask] = 0

        index = s_map.argsort()[-1]
        data[index] = 1
        data = torch.Tensor(data).unsqueeze(0)
        return data
    
    #@torchsnooper.snoop()
    def attack(self, x, true_y):
        batch_x = catch_batch_samples(self.set_x, self.batchsize)
        x.requires_grad = True
        batch_x[0] = x 
        if self.is_cuda:
            batch_x = batch_x.cuda()
        batch_out = self.model(batch_x)
        out = batch_out[0].unsqueeze(0)
        init_pred = out.cpu().max(1, keepdim=True)[1]
        if init_pred.item() != true_y.item(): 
            return 0, x.cuda() if self.is_cuda else x.cpu()

        for i in range(self.max_bit):
            self.model.zero_grad()
            out[0][0].backward(retain_graph=True)
            data_grad_0 = x.grad.data.clone()
            out[0][1].backward()
            data_grad_1 = x.grad.data.clone()
            data_grad = (data_grad_0.cpu(), data_grad_1.cpu())

            adv_x = self.generate(x.cpu(), data_grad, true_y).cuda() if self.is_cuda else self.generate(x.cpu(), data_grad, true_y)
            x = adv_x

            batch_x = batch_x.detach()
            x.requires_grad = True
            batch_x[0] = x
            batch_out = self.model(batch_x)
            out = batch_out[0].unsqueeze(0)

            adv_pred = out.cpu().max(1, keepdim=True)[1]
            if adv_pred.item() != init_pred.item():
                return i+1, adv_x.cuda() if self.is_cuda else adv_x.cpu()

        return -1, adv_x.cuda() if self.is_cuda else adv_x.cpu()