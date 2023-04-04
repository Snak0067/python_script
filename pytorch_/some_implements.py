# -*- coding:utf-8 -*-
# @FileName :some_implements.py
# @Time :2023/4/3 16:48
# @Author :Xiaofeng
import torch
from torch.autograd import Variable


def tensors_test():
    """
    PyTorch张量就像numpy数组一样,但它们可以在GPU上运行。
    没有计算图、梯度或深度学习的内置概念。
    在这里,我们使用PyTorch张量拟合一个两层网络
    """
    # 将程序运行在CPU上
    # dtype = torch.FloatTensor
    # 运行到GPU上,只需torch.cuda.FloatTensor
    dtype = torch.cuda.FloatTensor
    N, D_in, H, D_out = 64, 1000, 100, 10

    # Create random tensors for data and weights
    x = torch.randn(N, D_in).type(dtype)
    y = torch.randn(N, D_out).type(dtype)
    w1 = torch.randn(D_in, H).type(dtype)
    w2 = torch.randn(H, D_out).type(dtype)
    learning_rate = 1e-6
    for t in range(5000):
        h = x.mm(w1)
        h_relu = h.clamp(min=0)
        y_pred = h_relu.mm(w2)
        loss = (y_pred - y).pow(2).sum()

        grad_y_pred = 2.0 * (y_pred - y)
        grad_w2 = h_relu.t().mm(grad_y_pred)
        grad_h_relu = grad_y_pred.mm(w2.t())
        grad_h = grad_h_relu.clone()
        grad_h[h < 0] = 0
        grad_w1 = x.t().mm(grad_h)
        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2
        print(loss)


def pytorch_aotugrad():
    dtype = torch.cuda.FloatTensor
    N, D_in, H, D_out = 64, 1000, 100, 10
    x = Variable(torch.randn(N, D_in), requires_grad=False)
    y = Variable(torch.randn(N, D_out), requires_grad=False)
    w1 = Variable(torch.randn(D_in, H), requires_grad=True)
    w2 = Variable(torch.randn(H, D_out), requires_grad=True)
    learning_rate = 1e-6
    for t in range(500):
        y_pred = x.mm(w1).clamp(min=0).mm(w2)
        loss = (y_pred - y).pow(2).sum()
        if w1.grad: w1.grad.data.zero_()
        if w2.grad: w2.grad.data.zero()
        loss.backward()
        w1.data -= learning_rate * w1.grad.data
        w2.data -= learning_rate * w2.grad.data


def torch_test():
    print(torch.cuda.is_available())

    num_gpu = 1
    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and num_gpu > 0) else "cpu")
    print(device)
    print(torch.cuda.get_device_name(0))
    print(torch.rand(3, 3).cuda())


if __name__ == '__main__':
    tensors_test()
