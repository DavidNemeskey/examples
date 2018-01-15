from __future__ import print_function
import math

import torch
import torch.nn as nn


class LstmCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LstmCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.w_i_i = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.w_h_i = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.w_i_f = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.w_h_f = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.w_i_g = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.w_h_g = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.w_i_o = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.w_h_o = nn.Parameter(torch.Tensor(hidden_size, hidden_size))

        if bias:
            self.b_i_i = nn.Parameter(torch.Tensor(hidden_size))
            self.b_h_i = nn.Parameter(torch.Tensor(hidden_size))
            self.b_i_f = nn.Parameter(torch.Tensor(hidden_size))
            self.b_h_f = nn.Parameter(torch.Tensor(hidden_size))
            self.b_i_g = nn.Parameter(torch.Tensor(hidden_size))
            self.b_h_g = nn.Parameter(torch.Tensor(hidden_size))
            self.b_i_o = nn.Parameter(torch.Tensor(hidden_size))
            self.b_h_o = nn.Parameter(torch.Tensor(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hidden):
        h_t, c_t = hidden

        i = input.matmul(self.w_i_i) + h_t.matmul(self.w_h_i)
        f = input.matmul(self.w_i_f) + h_t.matmul(self.w_h_f)
        g = input.matmul(self.w_i_g) + h_t.matmul(self.w_h_g)
        o = input.matmul(self.w_i_o) + h_t.matmul(self.w_h_o)

        if self.bias:
            i += self.b_i_i + self.b_h_i
            f += self.b_i_f + self.b_h_f
            g += self.b_i_g + self.b_h_g
            o += self.b_i_o + self.b_h_o

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_t = f * c_t + i * g
        h_t = o * torch.tanh(c_t)

        return h_t, c_t


class LstmCell2(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LstmCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.w_i = nn.Parameter(torch.Tensor(input_size, 4 * hidden_size))
        self.w_h = nn.Parameter(torch.Tensor(hidden_size, 4 * hidden_size))

        if bias:
            self.b_i = nn.Parameter(torch.Tensor(4 * hidden_size))
            self.b_h = nn.Parameter(torch.Tensor(4 * hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hidden):
        h_t, c_t = hidden

        ifgo = input.matmul(self.w_i) + h_t.matmul(self.w_h)

        if self.bias:
            ifgo += self.b_i + self.b_h

        i, f, g, o = ifgo.chunk(4, 1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_t = f * c_t + i * g
        h_t = o * torch.tanh(c_t)

        return h_t, c_t
