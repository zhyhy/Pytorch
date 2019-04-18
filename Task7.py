import torch
from torch import nn,optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision import datasets
import os

BATCH_SIZE = 200
LR = 1e-2
EPOCH = 20
DOWNLOAD_MNIST = False

# -------------------------
#          加载数据
# -------------------------
if not(os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
# 不存在 mnist 或者 mnist 是空的
    DOWNLOAD_MNIST = True
train_dataset = datasets.MNIST(
    root = './mnist',
    train = True,   # 下载训练数据集
    transform = torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)
test_dataset = datasets.MNIST(
    root = './mnist',
    train = False,   # 下载训练数据集
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)

# 按照 batch_size 封装成 Tensor，然后只需在包装成 Variable 既可作为模型输入
train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv = nn.Sequential(                                  # input (batch, 1,28,28)
            nn.Conv2d(1,6,kernel_size=3,stride=1,padding=1),  # 图像大小：- > (batch, 6, 28 ,28)
            nn.ReLU(),
            nn.MaxPool2d(2),                                   # - > (batch, 6 ,14,14)
            nn.Conv2d(6,16,5,1,0),                             # - > (bacth, 16,10,10)
            nn.ReLU(),
            nn.MaxPool2d(2),                                   # - > (batch, 16,5,5)
        )
        self.fc = nn.Sequential(            # 全连接层
            nn.Linear(16*5*5,120),
            nn.Linear(120,84),
            nn.Linear(84,10)
        )
    def forward(self, x):
        x = self.conv(x)            # out.shape    ( batch, 16,5,5)
        x = x.view(x.size(0),-1)
        # view()函数的功能根reshape类似，用来转换size大小。x = x.view(batchsize, -1)中batchsize指转换后有几行，
        # 而-1指在不告诉函数有多少列的情况下，根据原tensor数据和batchsize自动分配列数。
        out = self.fc(x)
        return out
cnn = CNN()
print(cnn)

loss_func = nn.CrossEntropyLoss()           # 多分类用交叉损失熵函数
optimzer = optim.Adam(cnn.parameters(),lr = LR)

for eopch in range(EPOCH):
    print('eopch{}'.format(eopch+1))
    print('*'*10)
    running_loss = 0.0
    running_acc = 0.0
    # 训练
    for i, data in enumerate(train_loader,1):
        im,label = data
        im = Variable(im)
        label = Variable(label)
        out = cnn(im)
        loss = loss_func(out,label)
        running_loss += loss.data[0] * label.size(0)
        _,pred = torch.max(out,1)
        num_correct = (pred == label).sum()
        acc = (pred == label).float().mean()
        running_acc += num_correct.data[0]

        optimzer.zero_grad()
        loss.backward()
        optimzer.step()
        if eopch % 500 == 0:
            print('Finish {} epoch,Loss:{:.6f},Acc:{:.6f}'.format(eopch + 1, running_loss / (len(train_dataset)), running_acc / len(train_dataset)))
