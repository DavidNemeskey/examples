from __future__ import print_function
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')  # noqa
import matplotlib.pyplot as plt

from lstm_cell import LstmCell


class Sequence(nn.Module):
    def __init__(self, lstm_cell=nn.LSTMCell):
        super(Sequence, self).__init__()
        self.lstm1 = lstm_cell(1, 51)
        self.lstm2 = lstm_cell(51, 51)
        self.linear = nn.Linear(51, 1)
        # for param in self.lstm1.parameters():
        #     print(param.data)

    def forward(self, input, future=0):
        outputs = []
        h_t = Variable(torch.zeros(input.size(0), 51).double(), requires_grad=False)
        c_t = Variable(torch.zeros(input.size(0), 51).double(), requires_grad=False)
        h_t2 = Variable(torch.zeros(input.size(0), 51).double(), requires_grad=False)
        c_t2 = Variable(torch.zeros(input.size(0), 51).double(), requires_grad=False)
        # print('H_T\n', h_t)
        # print('H_T2\n', h_t2)
        if input.is_cuda:
            h_t = h_t.cuda()
            c_t = c_t.cuda()
            h_t2 = h_t2.cuda()
            c_t2 = c_t2.cuda()

        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        # if we should predict the future
        for i in range(future):
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        stacked = torch.stack(outputs, 1)
        outputs = stacked.squeeze(2)
        return outputs


class SequenceLSTM(nn.Module):
    def __init__(self):
        super(SequenceLSTM, self).__init__()
        self.lstm = nn.LSTM(1, 51, num_layers=2, batch_first=True)
        self.linear = nn.Linear(51, 1)

    def forward(self, input, future=0):
        # print('input', input.size())
        h_t = Variable(torch.zeros(2, input.size(0), 51).double(), requires_grad=False)
        c_t = Variable(torch.zeros(2, input.size(0), 51).double(), requires_grad=False)
        if input.is_cuda:
            h_t = h_t.cuda()
            c_t = c_t.cuda()
        # print('h_t', h_t.size())

        output, (h_t, c_t) = self.lstm(input.unsqueeze(2), (h_t, c_t))
        outputs = [self.linear(output)]
        # if we should predict the future
        # print('outputs', outputs[-1].size())
        output = outputs[-1][:, -1, :].unsqueeze(2)
        # print('output', output.size())
        for i in range(future):
            # print('future', i, output.size())
            output, (h_t, c_t) = self.lstm(output, (h_t, c_t))
            output = self.linear(output)
            # print('foutput', output.size())
            outputs += [output]
        # print('outputs:', end=' ')
        # for o in outputs:
        #     print(o.size(), end=' ')
        # print()
        catted = torch.cat(outputs, dim=1)
        outputs = catted.squeeze(2)
        # print('after cat:', catted.size(), 'after squeeze:', outputs.size())
        return outputs


if __name__ == '__main__':
    t = time.time()
    # set random seed to 0
    np.random.seed(0)
    torch.manual_seed(0)
    # load data and make training set
    data = torch.load('traindata.pt')
    input = Variable(torch.from_numpy(data[3:, :-1]), requires_grad=False)
    target = Variable(torch.from_numpy(data[3:, 1:]), requires_grad=False)
    test_input = Variable(torch.from_numpy(data[:3, :-1]), requires_grad=False)
    test_target = Variable(torch.from_numpy(data[:3, 1:]), requires_grad=False)
    # build the model
    # seq = Sequence(nn.LSTMCell)
    seq = Sequence(LstmCell)
    seq.double()
    # print('INPUT\n', input)
    if torch.cuda.is_available():
        input = input.cuda()
        target = target.cuda()
        test_input = test_input.cuda()
        test_target = test_target.cuda()
        seq = seq.cuda()
    criterion = nn.MSELoss()
    # use LBFGS as optimizer since we can load the whole data to train
    optimizer = optim.LBFGS(seq.parameters(), lr=0.8)
    # begin to train
    for i in range(15):
    # for i in range(1):
        print('STEP: ', i)

        def closure():
            optimizer.zero_grad()
            out = seq(input)
            # print('OUTPUT\n', out)
            loss = criterion(out, target)
            # print('LOSS\n', loss)
            print('loss:', loss.data.cpu().numpy()[0])
            loss.backward()
            return loss

        optimizer.step(closure)
        # begin to predict
        future = 1000
        pred = seq(test_input, future=future)
        loss = criterion(pred[:, :-future], test_target)
        print('test loss:', loss.data.cpu().numpy()[0])
        y = pred.data.cpu().numpy()
        continue
        # draw the result
        plt.figure(figsize=(30, 10))
        plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
        plt.xlabel('x', fontsize=20)
        plt.ylabel('y', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        def draw(yi, color):
            plt.plot(np.arange(input.size(1)), yi[:input.size(1)], color, linewidth=2.0)
            plt.plot(np.arange(input.size(1), input.size(1) + future), yi[input.size(1):], color + ':', linewidth=2.0)

        draw(y[0], 'r')
        draw(y[1], 'g')
        draw(y[2], 'b')
        plt.savefig('predict%d.pdf' % i)
        plt.close()
    print('Took {} seconds.'.format(time.time() - t))
