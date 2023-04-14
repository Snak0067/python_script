# -*- coding:utf-8 -*-
# @FileName :pytorch1.py
# @Time :2023/4/12 16:06
# @Author :Xiaofeng
import torch
import torch.nn as nn
import torch.optim as optim
from torch import dtype
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T

import numpy as np
import torch.nn.functional as F  # useful stateless functions


def flatten(x):
    N = x.shape[0]  # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image


def test_flatten():
    x = torch.arange(12).view(2, 1, 3, 2)
    print('Before flattening: ', x)
    print('After flattening: ', flatten(x))


def two_layer_fc(x, params):
    """
    A fully-connected neural networks; the architecture is:
    NN is fully connected -> ReLU -> fully connected layer.
    Note that this function only defines the forward pass;
    PyTorch will take care of the backward pass for us.

    The input to the network will be a minibatch of data, of shape
    (N, d1, ..., dM) where d1 * ... * dM = D. The hidden layer will have H units,
    and the output layer will produce scores for C classes.

    Inputs:
    - x: A PyTorch Tensor of shape (N, d1, ..., dM) giving a minibatch of
      input data.
    - params: A list [w1, w2] of PyTorch Tensors giving weights for the network;
      w1 has shape (D, H) and w2 has shape (H, C).

    Returns:
    - scores: A PyTorch Tensor of shape (N, C) giving classification scores for
      the input data x.
    """
    # first we flatten the image
    x = flatten(x)  # shape: [batch_size, C x H x W]

    w1, w2 = params

    # Forward pass: compute predicted y using operations on Tensors. Since w1 and
    # w2 have requires_grad=True, operations involving these Tensors will cause
    # PyTorch to build a computational graph, allowing automatic computation of
    # gradients. Since we are no longer implementing the backward pass by hand we
    # don't need to keep references to intermediate values.
    # you can also use `.clamp(min=0)`, equivalent to F.relu()
    x = F.relu(x.mm(w1))
    x = x.mm(w2)
    return x


def two_layer_fc_test():
    hidden_layer_size = 42
    x = torch.zeros(64, 50)  # minibatch size 64, feature dimension 50
    w1 = torch.zeros(50, hidden_layer_size)
    w2 = torch.zeros(hidden_layer_size, 10)
    scores = two_layer_fc(x, [w1, w2])
    # print(x.type())
    print(scores.size())  # you should see [64, 10]


def three_layer_convnet(x, params):
    """
    Performs the forward pass of a three-layer convolutional network with the
    architecture defined above.

    Inputs:
    - x: A PyTorch Tensor of shape (N, 3, H, W) giving a minibatch of images
    - params: A list of PyTorch Tensors giving the weights and biases for the
      network; should contain the following:
      - conv_w1: PyTorch Tensor of shape (channel_1, 3, KH1, KW1) giving weights
        for the first convolutional layer
      - conv_b1: PyTorch Tensor of shape (channel_1,) giving biases for the first
        convolutional layer
      - conv_w2: PyTorch Tensor of shape (channel_2, channel_1, KH2, KW2) giving
        weights for the second convolutional layer
      - conv_b2: PyTorch Tensor of shape (channel_2,) giving biases for the second
        convolutional layer
      - fc_w: PyTorch Tensor giving weights for the fully-connected layer. Can you
        figure out what the shape should be?
      - fc_b: PyTorch Tensor giving biases for the fully-connected layer. Can you
        figure out what the shape should be?

    Returns:
    - scores: PyTorch Tensor of shape (N, C) giving classification scores for x
    """
    conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b = params
    scores = None
    ################################################################################
    # TODO: Implement the forward pass for the three-layer ConvNet.                #
    ################################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ################################################################################
    #                                 END OF YOUR CODE                             #
    ################################################################################
    return scores


def torch_nn_Maxpool():
    input = torch.randn(20, 3, 32, 32)
    a = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=5, padding=2, bias=True)
    b = nn.ReLU()
    c = nn.MaxPool2d(kernel_size=5, padding=2)
    d = nn.Conv2d(in_channels=96, out_channels=48, kernel_size=3, padding=1, bias=True)
    e = nn.MaxPool2d(kernel_size=3, padding=1)
    a_out = a(input)
    b_out = b(a_out)
    c_out = c(b_out)
    d_out = d(c_out)
    e_out = e(d_out)
    print(input.shape)
    print(a_out.shape)
    print(b_out.shape)
    print(c_out.shape)
    print(d_out.shape)
    print(e_out.shape)


def conv(input, kernel_size, padding, stride):
    return (input - kernel_size + 2 * padding) // stride + 1


if __name__ == '__main__':
    a = conv(32, 5, 2, 1)
    b = conv(a, 3, 1, 1)
    c = conv(b, 3, 1, 2)
    d = conv(c, 3, 1, 1)
    e = conv(d, 2, 1, 2)
    f = conv(e, 3, 1, 1)
    g = conv(f, 2, 1, 1)
    h = conv(g, 2, 1, 2)

    print(h)
