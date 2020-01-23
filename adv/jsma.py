import torch
import numpy as np


class JSMA:
    def __init__(self, model, max_bit=10, is_cuda=False):
        self.model = model
        self.max_bit = max_bit
        self.is_cuda = is_cuda
        self.model.eval()
        if self.is_cuda:
            self.model = self.model.cuda()
    
    def generate(self, data, data_grad):
        data = data.data.numpy().squeeze()
        data_grad_0 = data_grad[0].numpy().squeeze()
        data_grad_1 = data_grad[1].numpy().squeeze()
        assert data.shape[0] == data_grad_0.shape[0] == data_grad_1.shape[0]

        # 计算saliency map. 我们的目标是将1->0, target=0,所以只需要S(X, 0)
        s_map_0 = np.zeros_like(data)
        for i in range(s_map_0.shape[0]):
            if data_grad_0[i] < 0 or data_grad_1[i] > 0:
                pass
            else:
                s_map_0[i] = data_grad_0[i] * abs(data_grad_1[i])
        
        # 根据saliency map 和 恶意软件限制条件 确定允许修改的bit
        #mask_data = np.array(list(map(lambda i: True if i==0 else False, data)))
        mask_data = (data == 0)
        #mask_smap = np.array(list(map(lambda x: True if x>0 else False, s_map_0)))
        mask_smap = (s_map_0 > 0)
        mask = mask_data & mask_smap
        s_map_0[~mask] = 0

        # 修改datar中的一个bit
        index = s_map_0.argsort()[-1]
        data[index] = 1
        data = torch.Tensor(data).unsqueeze(0)
        return data
    
    def attack(self, id, x, true_y):
        """
        return: 0表示本来模型就误判，不需要生成对抗样本。 -1表示对抗样本生成失败。 n(n>0)表示经过n次迭代成功生成对抗样本。
        """
        if self.is_cuda:
            x = x.cuda()
        x.requires_grad = True
        out = self.model.predict(id, x)
        init_pred = out.cpu().max(1, keepdim=True)[1]
        if init_pred.item() != true_y.item(): # 本来模型就判错的那些就不用管了。
            #print("no need! init_pred={}, true_y={}".format(init_pred.item(), true_y.item()))
            return 0, None

        for i in range(self.max_bit):
            self.model.zero_grad()
            out[0][0].backward(retain_graph=True)
            data_grad_0 = x.grad.data.clone()
            out[0][1].backward()
            data_grad_1 = x.grad.data.clone()
            data_grad = (data_grad_0.cpu(), data_grad_1.cpu())

            adv_x = self.generate(x.cpu(), data_grad)

            x = adv_x
            x.requires_grad = True
            out = self.model.predict(id, x)
            adv_pred = out.cpu().max(1, keepdim=True)[1]
            if adv_pred.item() != init_pred.item():
                #print("success! {} iters".format(i+1))
                return i+1, adv_x.cpu().data.numpy().squeeze() # 返回修改的bit数

        #print("failed! max up {} iters".format(i+1))
        return -1, None
