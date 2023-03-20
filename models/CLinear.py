import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    """
    Just one Linear layer
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = 7328
        self.pred_len = configs.pred_len

        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.channels = configs.enc_in
        self.individual = configs.individual
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len,self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        temp_2 = torch.einsum('abd, acd->abcd', x, x)
        temp_3 = torch.einsum('abce, ade->abcde', temp_2, x)
        x = [x]
        # x = torch.cat([x, torch.flatten(torch.einsum('abd, acd->abcd', x, x), start_dim=-3, end_dim=-2)], dim=-2)
        for i in range(x[0].shape[1]):
            x.append(temp_2[:, i, i:])
        for i in range(x[0].shape[1]):
            for j in range(i, x[0].shape[2]):
                x.append(temp_3[:, i, j, j:])
        x = torch.cat(x, dim=-2)
        if self.individual:
            output = torch.zeros([x.size(0),self.pred_len,x.size(2)],dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:,:,i] = self.Linear[i](x[:,:,i])
            x = output
        else:
            x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
        return x # [Batch, Output length, Channel]