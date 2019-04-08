# # import numpy as np
# #
# #
# # def func(x, y):
# #     return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2
# #
# #
# # # 函数对x求导
# # def dz_dx(x, y):
# #     return 2 * x - 400 * (y - x ** 2) * x - 2
# #
# #
# # # 函数对y求导
# # def dz_dy(x, y):
# #     return 200 * (y - x ** 2)
# #
# #
# # value = np.zeros(2)
# # learinng_rate = 0.001
# # loss = 10.0
# # iter_count = 0
# #
# # while loss > 0.001 and iter_count < 10000:
# #     error = np.zeros(2)
# #     error[0] = dz_dx(value[0], value[1])
# #     error[1] = dz_dy(value[0], value[1])
# #
# #     # x、y更新
# #     for i in range(2):
# #         value[i] = value[i] - learinng_rate * error[i]
# #     loss = func(value[0], value[1])
# #     print('迭代次数', iter_count, '损失：', loss)
# #     iter_count += 1
# #
# # import torch
# # from torch.autograd import Variable
# # import torch.nn as nn
# # import torch.nn.functional as f
# # import torch.optim as optim
# #
# # NUM = 100
# # hide_num = 300
# #
# #
# # class Net(nn.Module):
# #     def __init__(self):
# #         super(Net, self).__init__()
# #
# #         self.fc1 = nn.Linear(NUM, hide_num)
# #         self.fc2 = nn.Linear(hide_num, NUM)
# #
# #     def forward(self, x):
# #         x = f.relu(self.fc1(x))
# #         x = self.fc2(x)
# #         return x
# #
# #
# # net = Net()
# # print(net)
# # for param in net.parameters():
# #     print(param.size())
# # x = torch.randn(NUM)
# # input = Variable(x)
# #
# # target = Variable(0.5 * x + 0.3)
# #
# # optimizer = optim.SGD(net.parameters(), lr=0.01)
# # loss_list = []
# # step = 500
# # for epoch in range(step):
# #     optimizer.zero_grad()
# #     out = net(input)
# #     loss = nn.MSELoss()(out, target)
# #     loss_list.append(loss)
# #     loss.backward()
# #     optimizer.step()
# #
# # print('loss 的层级:')
# # t = loss.grad_fn
# # while t:
# #     print(t)
# #     t = t.next_functions[0][0]
# #
# # print('target:', target)
# # out1 = net(input)
# # print('out1:', out1)
# # print(nn.MSELoss()(out1, target))
#
#
# import numpy as np
#
#
# def weightsUpdate(data, w, b, learning_rate=0.01):
#     for x0, y0 in data:
#         y = np.dot(x0, w) + b
#         w_gradient = (y - y0) * x0.T
#         b_gradient = (y - y0)[0]
#         w -= w_gradient * learning_rate
#         b -= b_gradient
#         loss = 0.5 * np.square(y - y0)
#     return [w, b, loss[0][0]]
#
#
# def generateData(w, b, dataNum=10):
#     data = []
#
#     for i in range(dataNum):
#         noise = np.random.randn(1) * 0.01
#         x0 = np.random.randn(1, w.shape[0])
#         y0 = np.dot(x0, w) + b + noise
#         x = [x0, y0]
#         data.append(x)
#     return data
#
#
# def linearRegressionTrain(data):
#     w0 = np.random.randn(data[0][0].shape[1], 1)
#     b0 = np.random.randn(1)
#     for i in range(1000):
#         w0, b0, loss = weightsUpdate(data, w0, b0, 0.01)
#         if (i % 100 == 0):
#             print(loss)
#
#     return [w0, b0]
#
#
# # y=2*x1+3*x2+1
# w = np.array([[2], [3], [4], [5]])
# b = np.array([1])
#
# data = generateData(w, b)
# w0, b0 = linearRegressionTrain(data)
# print(" w=", w, '\n', "w0=", w0, '\n', "b=", b, '\n', "b0=", b0)
#
#
# import torch
# from torch.autograd import Variable
# import torch.nn as nn
#
# #生成数据
# x = torch.unsqueeze(torch.linspace(-1, 1, 200), dim = 1)
# y = 5 * x + 0.8 * torch.rand(x.size())
#
# X = Variable(x)
# Y = Variable(y)
#
# def init_parameters():
#     W = Variable( torch.randn(1, 1), requires_grad=True)
#     b = Variable( torch.zeros(1, 1), requires_grad=True )
#     parameters = {"W": W, "b": b}
#     return parameters
#
# def model(X, parameters):
#     return X * parameters["W"] + parameters["b"]
#
# def square_loss(y_hat, Y):
#     loss = (y_hat - Y).pow(2).sum()
#     return loss
#
# def update_parameters(parameters, lr):
#     parameters["W"].data -= lr * parameters["W"].grad.data
#     parameters["b"].data -= lr * parameters["b"].grad.data
#     return
#
#
# EPOCH = 100 # 迭代次数
# learning_rate = 0.001 # 学习速率
#
# parameters = init_parameters() # 参数初始化
#
# for t in range(EPOCH):
#     y_hat = model(X, parameters)
#     loss = square_loss(y_hat, Y)
#     loss.backward()
#     update_parameters(parameters, learning_rate)
#     if (t+1) % 20 == 0:
#         print(loss)
#     parameters["W"].grad.data.zero_()
#     parameters["b"].grad.data.zero_()
#
# print("参数\t", parameters["W"])
# print("常数项\t" , parameters["b"])