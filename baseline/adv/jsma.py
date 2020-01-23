import torch
import numpy as np
import torchsnooper

class JSMA:
    def __init__(self, model, max_bit=10, is_cuda=False):
        self.max_bit = max_bit
        self.is_cuda = is_cuda
        self.model = model
        self.model.eval()
        if self.is_cuda:
            self.model = self.model.cuda()

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
    def attack(self, input, true_y):
        x = input.cuda() if self.is_cuda else input
        x.requires_grad = True
        out = self.model(x)
        init_pred = out.cpu().max(1, keepdim=True)[1]
        if init_pred.item() != true_y.item():
            return 0, x.detach().cuda() if self.is_cuda else x.detach().cpu()

        for i in range(self.max_bit):
            self.model.zero_grad()
            out[0][0].backward(retain_graph=True)
            data_grad_0 = x.grad.data.clone()
            out[0][1].backward()
            data_grad_1 = x.grad.data.clone()
            data_grad = (data_grad_0.cpu(), data_grad_1.cpu())

            adv_x = self.generate(x.cpu(), data_grad, true_y).cuda() if self.is_cuda else self.generate(x.cpu(), data_grad, true_y)
            
            x = adv_x
            x.requires_grad = True
            out = self.model(x)
            adv_pred = out.cpu().max(1, keepdim=True)[1]
            if adv_pred.item() != init_pred.item():
                return i+1, adv_x.detach().cuda() if self.is_cuda else adv_x.detach().cpu()

        return -1, adv_x.detach().cuda() if self.is_cuda else adv_x.detach().cpu()
        