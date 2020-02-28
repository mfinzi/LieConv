#implementation adapted from https://github.com/vacancy/Synchronized-BatchNorm-PyTorch/issues/14

import torch
import torch.nn as nn


class MaskBatchNormNd(nn.BatchNorm1d):
    """ n-dimensional batchnorm that excludes points outside the mask from the statistics"""
    def forward(self, inp):
        """input _, (*, c), (*,) computes statistics averaging over * within the mask"""
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