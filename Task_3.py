# pytorch 实现逻辑回归
# 数据集 MNIST
import torch
import random
import numpy as np
import torch.nn as nn
import torchvision
import torchvision.transforms as transform
import torch.nn.functional as F

# 设置超参数
input_size = 784
num_class = 10
num_epochs = 10
batch_size = 100
lr = 0.001

# load data
train_dataset = torchvision.datasets.MNIST(root = './data/mnist',train=True,transform=transform.ToTensor(),download=True)
test_dataset = torchvision.datasets.MNIST(root = './data/mnist',train=False,transform=transform.ToTensor())

#torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, collate_fn=<function default_collate>, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None)
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,batch_size=batch_size,shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,batch_size = batch_size,shuffle = False)

class LR(nn.Module):
    def __init__(self,input_dims,output_dims):
        super().__init__()
        self.linear = nn.Linear(input_dims,output_dims,bias=True)
    def forward(self, x):
        x = self.linear(x)
        return x
LR_Module = LR(input_size,num_class)

criterion = nn.CrossEntropyLoss(reduction='mean')

optimer = torch.optim.SGD(LR_Module.parameters(),lr = lr)

cuda_gpu = torch.cuda.is_available()

step = len(train_loader)
for epoch in range(num_epochs):
    for i ,(images,labels) in enumerate(train_loader):
        if cuda_gpu:
            images, labels = images.cuda(), labels.cuda()
            LR_Module.cuda()
        # 图像size转换为(batch_size,input_size),应该为（100,784,）
        images = images.reshape(-1,28*28)

        # 前向传播
        y_pred = LR_Module(images)
        loss = criterion(y_pred,labels)

        # 反向传播
        optimer.zero_grad()
        loss.backward()
        optimer.step()
        if(i % 100 ==0):
             print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, step, loss.item()))

# 测试阶段
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        if cuda_gpu:
            images, labels = images.cuda(), labels.cuda()
            LR_Module.cuda()
        images = images.reshape(-1,28*28)
        output = LR_Module(images)

        max , pred = torch.max(output.data,1)
        total += labels.size(0)
        correct += (pred == labels).sum()

    print('Acc of the model is :{} %  ' .format(100 * correct / total))


