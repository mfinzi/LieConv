#implementation adapted from https://github.com/vacancy/Synchronized-BatchNorm-PyTorch/issues/14

import torch
import torch.nn as nn


class MaskBatchNormNd(nn.BatchNorm1d):

    def forward(self, inp):
        """input _, (*, c), (*,) computes statistics averaging over * excluding mask"""
        coords,x,mask = inp
        sum_dims = list(range(len(x.shape[:-1])))
        x_or_zero = torch.where(mask.unsqueeze(-1),x,torch.zeros_like(x)) #remove nans
        if self.training or not self.track_running_stats:
            xsum = x_or_zero.sum(dim=sum_dims)
            xxsum = (x_or_zero*x_or_zero).sum(dim=sum_dims)
            numel_notnan = (mask).sum()
            xmean = xsum / numel_notnan
            sumvar = xxsum - xsum * xmean
            unbias_var = sumvar / (numel_notnan - 1)
            bias_var = sumvar / numel_notnan
            self.running_mean = (
                (1 - self.momentum) * self.running_mean
                + self.momentum * xmean.detach())
            self.running_var = (
                (1 - self.momentum) * self.running_var
                + self.momentum * unbias_var.detach())
        else:
            xmean, bias_var = self.running_mean,self.running_var
        std = bias_var.clamp(self.eps) ** 0.5
        ratio = self.weight/std
        output = (x_or_zero*ratio + (self.bias - xmean*ratio))
        return (coords,output,mask)

class NanBatchNorm(nn.Module):

    def __init__(
            self,
            num_features,
            eps=1e-5,
            momentum=0.1):
        super(NanBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.Tensor(num_features))
        self.bias = nn.Parameter(torch.Tensor(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()

    def reset_running_stats(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)

    def reset_parameters(self):
        self.reset_running_stats()
        nn.init.uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, input_):
        batchsize, channels, height, width = input_.size()
        numel = batchsize * height * width
        numel_notnan = torch.logical_not(torch.isnan(input_[:, 0])).sum()
        input_ = input_.permute(1, 0, 2, 3).contiguous().view(channels, numel)
        nan_mask = torch.isnan(input_)
        input_[nan_mask] = 0. # this way nans do not affect statistics
        sum_ = input_.sum(1)
        sum_of_square = input_.pow(2).sum(1)
        mean = sum_ / numel_notnan
        sumvar = sum_of_square - sum_ * mean
        unbias_var = sumvar / (numel_notnan - 1)
        bias_var = sumvar / numel_notnan
        std = bias_var.clamp(self.eps) ** 0.5

        self.running_mean = (
            (1 - self.momentum) * self.running_mean
            + self.momentum * mean.detach())
        self.running_var = (
            (1 - self.momentum) * self.running_var
            + self.momentum * unbias_var.detach())

        output = (
            (input_ - mean.unsqueeze(1)) / std.unsqueeze(1) *
            self.weight.unsqueeze(1) + self.bias.unsqueeze(1))
        output[nan_mask] = float('nan')

        return output.view(channels, batchsize, height, width).permute(1,0,2,3).contiguous()