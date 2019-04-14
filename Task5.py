import torch
from torch import nn as nn
from torch.autograd import Variable
from torch.nn import functional as F


# class MLP(nn.Module):
#     def __init__(self):
#         super(MLP,self).__init__()
#         self.L1 = nn.Linear(64,32)
#         self.L2 = nn.Linear(32,16)
#         self.L3 = nn.Linear(16,2)
#
#     def forward(self,x):
#         lay1_out = F.relu(self.L1(x))
#         lay2_out = F.relu(self.L2(x))
#         out = self.L3(x)
#         return out,lay1_out,lay2_out
#
# def l1_re(var):
#     return torch.abs(var).sum()
#
# def l2_re(var):
#     return torch.sqrt(torch.pow(var,2).sum())
#
# BATCH_SIZE = 10
#
# lambda1, labmda2 =  0.5,0.1
# for i in range(100):
#     mlp = MLP()
#     opt = torch.optim.SGD(mlp.parameters(),lr = 1e-4)
#
#     x = Variable(torch.rand(BATCH_SIZE,64))
#     y = Variable(torch.ones(BATCH_SIZE).long())
#
#     opt.zero_grad()
#     pre,lay1,lay2 = mlp(x)
#     corss_entropy_loss = F.cross_entropy(pre,y)
#     l1_regularization = lambda1 * l1_re(lay1)
#     l2_regularization = labmda2 * l2_re(lay2)
#     loss = corss_entropy_loss + l1_regularization + l2_regularization
#     loss.backward()
#     opt.step()


#     ''' dropout '''
# hidden_num = 128
# out_num = 10
#
# class Net(nn.Module):
#     def __init__(self,x_in,y_out,num_hidden):
#         super(Net,self).__init__()
#         self.lay1 = nn.Linear(x_in,num_hidden,bias=True)
#         self.lay2 = nn.Linear(num_hidden,y_out,bias=True)
#
#     def forward(self,x):
#         x = x.view(x.size(0),-1)
#         x = F.relu(self.lay1(x))
#         x = F.relu(self.lay2(x))
#         x = F.dropout(x,p = 0.5)
#         x = self.lay2(x)
#         return F.log_softmax(x,dim=1)






