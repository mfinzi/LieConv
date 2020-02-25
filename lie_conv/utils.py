import numpy as np
import torch
import torch.nn as nn
import sys
import random


def export(fn):
    mod = sys.modules[fn.__module__]
    if hasattr(mod, '__all__'):
        mod.__all__.append(fn.__name__)
    else:
        mod.__all__ = [fn.__name__]
    return fn
export(export)

@export
class Named(type):
    def __str__(self):
        return self.__name__
    def __repr__(self):
        return self.__name__

@export
class Expression(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func
        
    def forward(self, x):
        return self.func(x)

@export
def conv2d(in_channels,out_channels,kernel_size=3,coords=False,dilation=1,**kwargs):
    """ Wraps nn.Conv2d and CoordConv, padding is set to same
        and coords=True can be specified to get additional coordinate in_channels"""
    assert 'padding' not in kwargs, "assumed to be padding = same "
    assert not coords, "coordconv not currently supported"
    same = (kernel_size//2)*dilation
    return nn.Conv2d(in_channels,out_channels,kernel_size,padding=same,dilation=dilation,**kwargs)

@export
class FixedNumpySeed(object):
    def __init__(self, seed):
        self.seed = seed
    def __enter__(self):
        self.np_rng_state = np.random.get_state()
        np.random.seed(self.seed)
        self.rand_rng_state = random.getstate()
        random.seed(self.seed)
    def __exit__(self, *args):
        np.random.set_state(self.np_rng_state)
        random.setstate(self.rand_rng_state)

@export
class RandomZrotation(nn.Module):
    def __init__(self,max_angle=np.pi):
        super().__init__()
        self.max_angle = max_angle
    def forward(self,x):
        if self.training:
            # this presumes z axis is coordinate 2?
            # assumes x has shape B3N
            bs,c,n = x.shape; assert c==3
            angles = (2*torch.rand(bs)-1)*self.max_angle
            R = torch.zeros(bs,3,3)
            R[:,2,2] = 1
            R[:,0,0] = R[:,1,1] = angles.cos()
            R[:,0,1] = R[:,1,0] = angles.sin()
            R[:,1,0] *=-1
            return R.to(x.device)@x
        else:
            return x
@export
class GaussianNoise(nn.Module):
    """ Layer that adds pixelwise gaussian noise to input (during train)"""
    def __init__(self, std):
        super().__init__()
        self.std = std
    
    def forward(self, x):
        if self.training:
            return x + torch.randn_like(x)*self.std
        else:
            return x