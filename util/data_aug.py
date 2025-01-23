import torch
from torch import nn
import numpy as np
from torch.autograd import Variable


class _DataAugmentation(nn.Module):
    """ Base Module for data augmentation techniques. """


class MixUp(_DataAugmentation):
    def __init__(self, alpha=0.3):
        super(MixUp, self).__init__()
        self.alpha = alpha
        self.lam = 1

    def forward(self, x, y):
        self.lam = np.random.beta(self.alpha, self.alpha)
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(x.device)

        x = self.lam * x + (1 - self.lam) * x[index, :]
        y_a, y_b = y, y[index]
        x, y_a, y_b = map(Variable, (x, y_a, y_b))
        return x, (y_a, y_b)


class MultiMixUp(_DataAugmentation):
    def __init__(self, alpha=0.3):
        super(MultiMixUp, self).__init__()
        self.alpha = alpha
        self.lam = 1

    def forward(self, x: list, y: list):
        self.lam = np.random.beta(self.alpha, self.alpha)
        batch_size = x[0].size()[0]
        index = torch.randperm(batch_size).to(x[0].device)
        x_out = []
        for x_i in x:
            x_out_i = self.lam * x_i + (1 - self.lam) * x_i[index, :]
            x_out.append(x_out_i)
        y_out = []
        for y_i in y:
            y_a, y_b = y_i, y_i[index]
            y_out.append((y_a, y_b))
        return x_out, y_out

    def loss_calculate(self, y_hat, y, loss_func):
        loss = self.lam * loss_func(y_hat, y[0]) + (1 - self.lam) * loss_func(y_hat, y[1])
        return loss

    def accuracy_calculate(self, pred, y):
        corrects = (self.lam * torch.sum(pred == y[0]) + (1 - self.lam) * torch.sum(pred == y[1]))
        acc = corrects.item() / len(pred)
        return acc