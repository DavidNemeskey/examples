from __future__ import print_function
from itertools import product
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')  # noqa
import matplotlib.pyplot as plt


class LinReg(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super(LinReg, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.bias = bias

        self.A = nn.Parameter(torch.Tensor(in_dim, out_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(out_dim))

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        Ax = torch.mm(x, self.A)
        return Ax + self.b if self.bias else Ax


def test1():
    x = np.arange(10, dtype=float).reshape(10, 1)
    y = x * 0.5 + 3
    train(x, y, 0.01, 100)


def test2():
    x = np.array(list(product(range(10), repeat=2)))
    y = x.dot([[0.5], [-2]]) + 3
    # y = Variable(x.data.mm(torch.Tensor([[0.5], [-2]])) + 3, requires_grad=False)
    train(x, y, 0.01, log_every=100)


def train(x, y, lr, iter=None, log_every=None):
    num_items = len(x)
    if not iter:
        iter = num_items * 50
    x = Variable(torch.Tensor(x), requires_grad=False)
    y = Variable(torch.Tensor(y), requires_grad=False)
    linreg = LinReg(x.shape[-1], y.shape[-1])

    criterion = nn.MSELoss()

    for i in range(iter):
        linreg.zero_grad()
        y_hat = linreg.forward(x[i % num_items])
        loss = criterion(y_hat, y[i % num_items])
        if log_every and i > 0 and i % log_every == 0:
            print('Loss at step {}: {}'.format(i, loss.data.numpy().sum()))
        loss.backward()
        for p in linreg.parameters():
            p.data.add_(-lr, p.grad.data)
    print('A: {}, b: {}'.format(linreg.A.data.numpy(), linreg.b.data.numpy()))


if __name__ == '__main__':
    test1()
    test2()
