#implementation adapted from https://github.com/vacancy/Synchronized-BatchNorm-PyTorch/issues/14

import torch
import torch.nn as nn
import torch.nn.init as init
from lie_conv.utils import export

@export
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
            numel_mask = (mask).sum()
            if mask.shape!=x.shape[:-1]:
                numel_mask *= x.shape[0]
            xmean = xsum / numel_mask
            sumvar = xxsum - xsum * xmean
            unbias_var = sumvar / (numel_mask - 1)
            bias_var = sumvar / numel_mask
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


# Ported from https://github.com/yechengxi/deconvolution/blob/master/models/deconv.py

#iteratively solve for inverse sqrt of a matrix
def isqrt_newton_schulz_autograd(A, numIters):
    dim = A.shape[0]
    normA=A.norm()
    Y = A.div(normA)
    I = torch.eye(dim,dtype=A.dtype,device=A.device)
    Z = torch.eye(dim,dtype=A.dtype,device=A.device)

    for i in range(numIters):
        T = 0.5*(3.0*I - Z@Y)
        Y = Y@T
        Z = T@Z
    #A_sqrt = Y*torch.sqrt(normA)
    A_isqrt = Z / torch.sqrt(normA)
    return A_isqrt

@export
class DecorrelateBN(nn.Module):
    def __init__(self,  channels, num_groups=None, eps=1e-4,n_iter=5,momentum=0.1,debug=False):
        super().__init__()

        self.eps = eps
        self.n_iter=n_iter
        self.momentum=momentum
        if num_groups==None: num_groups=channels
        self.num_groups = num_groups
        self.debug=debug
        
        self.register_buffer('running_mean1', torch.zeros(num_groups, 1))
        #self.register_buffer('running_cov', torch.eye(num_groups))
        self.register_buffer('running_deconv', torch.eye(num_groups))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        self.weight = nn.Parameter(torch.empty(channels-channels%num_groups))
        self.bias = nn.Parameter(torch.empty(channels-channels%num_groups))
        self.reset_parameters()
        if channels%num_groups!=0:
            self.regular_masked_bn = MaskBatchNormNd(channels%num_groups)
    def reset_parameters(self):
        init.uniform_(self.weight)
        init.zeros_(self.bias)
    def forward(self, inp):
        """input _, (*, c), (*,) computes statistics averaging over * within the mask"""
        coords,x,mask = inp
        x_or_zero = torch.where(mask.unsqueeze(-1),x,torch.zeros_like(x)) #remove nans
        orig_shape = x.shape
        
        x_flat = x_or_zero.view(-1,x.shape[-1])
        numel_mask = (mask).sum()
        if mask.shape!=x.shape[:-1]:
            numel_mask *= x.shape[0]

        N, C = x_flat.size()
        G = self.num_groups
        
        #take the first c channels out for deconv
        c=int(C/G)*G
        if c==0:
            print('Error! num_groups should be set smaller.')

        #step 1. remove mean
        if c!=C:
            x1=x_flat[:,:c].permute(1,0).contiguous().view(G,-1)
        else:
            x1=x_flat.permute(1,0).contiguous().view(G,-1)

        #step 1. remove mean
        #assert c==C, f"chs:{C} should be a multiple of G:{G}"
        flat_mask = mask.expand(x.shape[:-1]).reshape(1,-1).repeat((c,1)).view(G,-1)
        G_numel = flat_mask.sum(-1)[0]

        mean1 = x1.sum(-1, keepdim=True)/G_numel

        if self.num_batches_tracked==0:
            self.running_mean1.copy_(mean1.detach())
        if self.training:
            self.running_mean1.mul_(1-self.momentum)
            self.running_mean1.add_(mean1.detach()*self.momentum)
        else:
            mean1 = self.running_mean1

        x1=torch.where(flat_mask,x1-mean1,torch.zeros_like(x1))

        #step 2. calculate deconv@x1 = cov^(-0.5)@x1
        if self.training:
            cov = x1 @ x1.t() / G_numel + self.eps * torch.eye(G, dtype=x.dtype, device=x.device)
            deconv = isqrt_newton_schulz_autograd(cov, self.n_iter)

        if self.num_batches_tracked==0:
            #self.running_cov.copy_(cov.detach())
            self.running_deconv.copy_(deconv.detach())

        if self.training:
            #self.running_cov.mul_(1-self.momentum)
            #self.running_cov.add_(cov.detach()*self.momentum)
            self.running_deconv.mul_(1 - self.momentum)
            self.running_deconv.add_(deconv.detach() * self.momentum)
        else:
            # cov = self.running_cov
            deconv = self.running_deconv

        x1 =deconv@x1

        #reshape to N,c,J,W
        x1 = x1.view(c, N).contiguous().permute(1,0)
        if self.training:
            self.num_batches_tracked.add_(1)
        x1 = (x1*self.weight.unsqueeze(0) + self.bias.unsqueeze(0)).view(x[...,:c].shape)
        if c!=C:
            _,x_leftover,_ =self.regular_masked_bn((coords,x[...,c:],mask))
            x1 = torch.cat([x1, x_leftover], dim=-1)
        
        return coords,x1,mask