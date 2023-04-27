""" 
Note to the Reader:

In order to handle the generic spatial data we process features as a tuple: (coordinates,values,mask)
with shapes (bs,n,d), (bs,n,c), (bs,n) where bs is batchsize, n is the maximum number of points in the minibatch,
d is the dimension of the coordinates, and c is the number of channels of the feature map.
The mask specifies which elements are valid and ~mask specifies the elements that have been added through padding.
For the PointConv operation and networks elements are passed through the network as this tuple (coordinates,values,mask).

Naively for LieConv we would process (lie_algebra_elems,values,mask) with the same shapes but with d as the dimension of the group.
However, as an optimization to avoid repeated calculation of the pairs log(v^{-1}u)=log(e^{-b}e^{a}), we instead compute this for all
pairs once at the lifting stage which has the name 'ab_pairs' in the code and shape (bs,n,n,d). Subsampling operates on this matrix
by subsampling both n axes. abq_pairs also includes the q pairs (embedded orbit identifiers) so abq_pairs =  [log(e^{-b}e^{a}),qa,qb].
So the tuple  (abq_pairs,values,mask) with shapes (bs,n,n,d) (bs,n,c) (bs,n) is passed through the network. 
The 'Pass' module is used extensively to operate on only one of these and return the tuple with the rest unchanged, 
such as for computing a swish nonlinearity on values.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lie_conv.utils import Expression,export,Named, Pass
from lie_conv.utils import FarthestSubsample, knn_point, index_points
from lie_conv.lieGroups import T,SO2,SO3,SE2,SE3, norm
from lie_conv.masked_batchnorm import MaskBatchNormNd


@export
def Swish():
    return Expression(lambda x: x*torch.sigmoid(x))

def LinearBNact(chin,chout,act='swish',bn=True):
    """assumes that the inputs to the net are shape (bs,n,mc_samples,c)"""
    assert act in ('relu','swish'), f"unknown activation type {act}"
    normlayer = MaskBatchNormNd(chout)
    return nn.Sequential(
        Pass(nn.Linear(chin,chout),dim=1),
        normlayer if bn else nn.Sequential(),
        Pass(Swish() if act=='swish' else nn.ReLU(),dim=1))

def WeightNet(in_dim,out_dim,act,bn,k=32):
    return nn.Sequential(
        *LinearBNact(in_dim, k, act, bn),
        *LinearBNact(k, k, act, bn),
        *LinearBNact(k, out_dim, act, bn))

class PointConv(nn.Module):
    def __init__(self,chin,chout,mc_samples=32,xyz_dim=3,ds_frac=1,knn_channels=None,act='swish',bn=False,mean=False):
        super().__init__()
        self.chin = chin # input channels
        self.cmco_ci = 16 # a hyperparameter controlling size and bottleneck compute cost of weightnet
        self.xyz_dim = xyz_dim # dimension of the space on which convolution operates
        self.knn_channels = knn_channels # number of xyz dims on which to compute knn
        self.mc_samples = mc_samples # number of samples to use to estimate convolution
        self.weightnet = WeightNet(xyz_dim, self.cmco_ci, act, bn) # MLP - final layer to compute kernel vals (see A1)
        self.linear = nn.Linear(self.cmco_ci * chin, chout)        # final linear layer to compute kernel vals (see A1)
        self.mean=mean  # Whether or not to divide by the number of mc_samples
        assert ds_frac==1, "test branch no ds, will need to carefully check that subsample respects padding"
        self.subsample = FarthestSubsample(ds_frac,knn_channels=knn_channels) # Module for subsampling if ds_frac<1

    def extract_neighborhood(self,inp,query_xyz):
        """ inputs shape ([inp_xyz (bs,n,d)], [inp_vals (bs,n,c)], [query_xyz (bs,m,d)])"""
        inp_xyz,inp_vals,mask = inp
        neighbor_idx = knn_point(min(self.mc_samples, inp_xyz.shape[1]),
                    inp_xyz[:,:,:self.knn_channels], query_xyz[:,:,:self.knn_channels],mask)
        neighbor_xyz = index_points(inp_xyz, neighbor_idx) # (bs,n,mc_samples,d)
        neighbor_values = index_points(inp_vals, neighbor_idx) #(bs,n,mc_samples,c)
        neighbor_mask = index_points(mask,neighbor_idx) # (bs,n,mc_samples)
        return neighbor_xyz, neighbor_values, neighbor_mask

    def point_convolve(self,embedded_group_elems,nbhd_vals, nbhd_mask):
        """ embedded_group_elems: (bs,m,nbhd,d)
            nbhd_vals: (bs,m,mc_samples,ci)
            nbhd_mask: (bs,m,mc_samples)"""
        bs, m, nbhd, ci = nbhd_vals.shape  # (bs,m,mc_samples,d) -> (bs,m,mc_samples,cm*co/ci)
        _, penult_kernel_weights, _ = self.weightnet((None,embedded_group_elems,nbhd_mask))
        penult_kernel_weights_m = torch.where(nbhd_mask.unsqueeze(-1),penult_kernel_weights,torch.zeros_like(penult_kernel_weights))
        nbhd_vals_m = torch.where(nbhd_mask.unsqueeze(-1),nbhd_vals,torch.zeros_like(nbhd_vals))
        #      (bs,m,mc_samples,ci) -> (bs,m,ci,mc_samples) @ (bs, m, mc_samples, cmco/ci) -> (bs,m,ci,cmco/ci) -> (bs,m, cmco) 
        partial_convolved_vals = (nbhd_vals_m.transpose(-1,-2)@penult_kernel_weights_m).view(bs, m, -1)
        convolved_vals = self.linear(partial_convolved_vals) #  (bs,m,cmco) -> (bs,m,co)
        if self.mean: convolved_vals /= nbhd_mask.sum(-1,keepdim=True).clamp(min=1)
        return convolved_vals

    def get_embedded_group_elems(self,output_xyz,nbhd_xyz):
        return output_xyz - nbhd_xyz
    
    def forward(self, inp):
        """inputs, and outputs have shape ([xyz (bs,n,d)], [vals (bs,n,c)])
            query_xyz has shape (bs,n,d)"""
        query_xyz, sub_vals, sub_mask = self.subsample(inp)
        nbhd_xyz, nbhd_vals, nbhd_mask = self.extract_neighborhood(inp, query_xyz)
        deltas = self.get_embedded_group_elems(query_xyz.unsqueeze(2), nbhd_xyz)
        convolved_vals = self.point_convolve(deltas, nbhd_vals, nbhd_mask)
        convolved_wzeros = torch.where(sub_mask.unsqueeze(-1),convolved_vals,torch.zeros_like(convolved_vals))
        return query_xyz, convolved_wzeros, sub_mask

def FPSindices(dists,frac,mask):
    """ inputs: pairwise distances DISTS (bs,n,n), downsample_frac (float), valid atom mask (bs,n) 
        outputs: chosen_indices (bs,m) """
    m = int(np.round(frac*dists.shape[1]))
    device = dists.device
    bs,n = dists.shape[:2]
    chosen_indices = torch.zeros(bs, m, dtype=torch.long,device=device)
    distances = torch.ones(bs, n,device=device) * 1e8
    a = torch.randint(0, n, (bs,), dtype=torch.long,device=device) #choose random start
    idx = a%mask.sum(-1) + torch.cat([torch.zeros(1,device=device).long(),torch.cumsum(mask.sum(-1),dim=0)[:-1]],dim=0)
    farthest = torch.where(mask)[1][idx]
    B = torch.arange(bs, dtype=torch.long,device=device)
    for i in range(m):
        chosen_indices[:, i] = farthest # add point that is farthest to chosen
        dist = torch.where(mask,dists[B,farthest],-100*torch.ones_like(distances)) # (bs,n) compute distance from new point to all others
        closer = dist < distances      # if dist from new point is smaller than chosen points so far
        distances[closer] = dist[closer] # update the chosen set's distance to all other points
        farthest = torch.max(distances, -1)[1] # select the point that is farthest from the set
    return chosen_indices


class FPSsubsample(nn.Module):
    def __init__(self,ds_frac,cache=False,group=None):
        super().__init__()
        self.ds_frac = ds_frac
        self.cache=cache
        self.cached_indices = None
        self.group = group
    def forward(self,inp,withquery=False):
        abq_pairs,vals,mask = inp
        dist = self.group.distance if self.group else lambda ab: norm(ab,dim=-1)
        if self.ds_frac!=1:
            if self.cache and self.cached_indices is None:
                query_idx = self.cached_indices = FPSindices(dist(abq_pairs),self.ds_frac,mask).detach()
            elif self.cache:
                query_idx = self.cached_indices
            else:
                query_idx = FPSindices(dist(abq_pairs),self.ds_frac,mask)
            B = torch.arange(query_idx.shape[0],device=query_idx.device).long()[:,None]
            subsampled_abq_pairs = abq_pairs[B,query_idx][B,:,query_idx]
            subsampled_values = vals[B,query_idx]
            subsampled_mask = mask[B,query_idx]
        else:
            subsampled_abq_pairs = abq_pairs
            subsampled_values = vals
            subsampled_mask = mask
            query_idx = None
        if withquery: return (subsampled_abq_pairs,subsampled_values,subsampled_mask, query_idx)
        return (subsampled_abq_pairs,subsampled_values,subsampled_mask)

class LieConv(PointConv):
    def __init__(self,*args,group=T(3),ds_frac=1,fill=1/3,cache=False,knn=False,**kwargs):
        kwargs.pop('xyz_dim',None)
        super().__init__(*args,xyz_dim=group.lie_dim+2*group.q_dim,**kwargs)
        self.group = group # Equivariance group for LieConv
        self.register_buffer('r',torch.tensor(2.)) # Internal variable for local_neighborhood radius, set by fill
        self.fill_frac = min(fill,1.) # Average Fraction of the input which enters into local_neighborhood, determines r
        self.knn=knn            # Whether or not to use the k nearest points instead of random samples for conv estimator
        self.subsample = FPSsubsample(ds_frac,cache=cache,group=self.group)
        self.coeff = .5  # Internal coefficient used for updating r
        self.fill_frac_ema = fill # Keeps track of average fill frac, used for logging only
        
    def extract_neighborhood(self,inp,query_indices):
        """ inputs: [pairs_abq (bs,n,n,d), inp_vals (bs,n,c), mask (bs,n), query_indices (bs,m)]
            outputs: [neighbor_abq (bs,m,mc_samples,d), neighbor_vals (bs,m,mc_samples,c)]"""

        # Subsample pairs_ab, inp_vals, mask to the query_indices
        pairs_abq, inp_vals, mask = inp
        if query_indices is not None:
            B = torch.arange(inp_vals.shape[0],device=inp_vals.device).long()[:,None]
            abq_at_query = pairs_abq[B,query_indices]
            mask_at_query = mask[B,query_indices]
        else:
            abq_at_query = pairs_abq
            mask_at_query = mask
        vals_at_query = inp_vals
        dists = self.group.distance(abq_at_query) #(bs,m,n,d) -> (bs,m,n)
        dists = torch.where(mask[:,None,:].expand(*dists.shape),dists,1e8*torch.ones_like(dists))
        k = min(self.mc_samples,inp_vals.shape[1])
        
        # Determine ids (and mask) for points sampled within neighborhood (A4)
        if self.knn: # NBHD: KNN
            nbhd_idx = torch.topk(dists,k,dim=-1,largest=False,sorted=False)[1] #(bs,m,nbhd)
            valid_within_ball = (nbhd_idx>-1)&mask[:,None,:]&mask_at_query[:,:,None]
            assert not torch.any(nbhd_idx>dists.shape[-1]), f"error with topk,\
                        nbhd{k} nans|inf{torch.any(torch.isnan(dists)|torch.isinf(dists))}"
        else: # NBHD: Sampled Distance Ball
            bs,m,n = dists.shape
            within_ball = (dists < self.r)&mask[:,None,:]&mask_at_query[:,:,None] # (bs,m,n)
            B = torch.arange(bs)[:,None,None]
            M = torch.arange(m)[None,:,None]
            noise = torch.zeros(bs,m,n,device=within_ball.device)
            noise.uniform_(0,1)
            valid_within_ball, nbhd_idx =torch.topk(within_ball+noise,k,dim=-1,largest=True,sorted=False)
            valid_within_ball = (valid_within_ball>1)
        
        # Retrieve abq_pairs, values, and mask at the nbhd locations
        B = torch.arange(inp_vals.shape[0],device=inp_vals.device).long()[:,None,None].expand(*nbhd_idx.shape)
        M = torch.arange(abq_at_query.shape[1],device=inp_vals.device).long()[None,:,None].expand(*nbhd_idx.shape)
        nbhd_abq = abq_at_query[B,M,nbhd_idx]     #(bs,m,n,d) -> (bs,m,mc_samples,d)
        nbhd_vals = vals_at_query[B,nbhd_idx]   #(bs,n,c) -> (bs,m,mc_samples,c)
        nbhd_mask = mask[B,nbhd_idx]            #(bs,n) -> (bs,m,mc_samples)
        
        if self.training and not self.knn: # update ball radius to match fraction fill_frac inside
            navg = (within_ball.float()).sum(-1).sum()/mask_at_query[:,:,None].sum()
            avg_fill = (navg/mask.sum(-1).float().mean()).cpu().item()
            self.r +=  self.coeff*(self.fill_frac - avg_fill)
            self.fill_frac_ema += .1*(avg_fill-self.fill_frac_ema)
        return nbhd_abq, nbhd_vals, (nbhd_mask&valid_within_ball.bool())

    # def log_data(self,logger,step,name):
    #     logger.add_scalars('info', {f'{name}_fill':self.fill_frac_ema}, step=step)
    #     logger.add_scalars('info', {f'{name}_R':self.r}, step=step)

    def point_convolve(self,embedded_group_elems,nbhd_vals,nbhd_mask):
        """ Uses generalized PointConv trick (A1) to compute convolution using pairwise elems (aij) and nbhd vals (vi).
            inputs [embedded_group_elems (bs,m,mc_samples,d), nbhd_vals (bs,m,mc_samples,ci), nbhd_mask (bs,m,mc_samples)]
            outputs [convolved_vals (bs,m,co)]"""
        bs, m, nbhd, ci = nbhd_vals.shape  # (bs,m,mc_samples,d) -> (bs,m,mc_samples,cm*co/ci)
        _, penult_kernel_weights, _ = self.weightnet((None,embedded_group_elems,nbhd_mask))
        penult_kernel_weights_m = torch.where(nbhd_mask.unsqueeze(-1),penult_kernel_weights,torch.zeros_like(penult_kernel_weights))
        nbhd_vals_m = torch.where(nbhd_mask.unsqueeze(-1),nbhd_vals,torch.zeros_like(nbhd_vals))
        #      (bs,m,mc_samples,ci) -> (bs,m,ci,mc_samples) @ (bs, m, mc_samples, cmco/ci) -> (bs,m,ci,cmco/ci) -> (bs,m, cmco) 
        partial_convolved_vals = (nbhd_vals_m.transpose(-1,-2)@penult_kernel_weights_m).view(bs, m, -1)
        convolved_vals = self.linear(partial_convolved_vals) #  (bs,m,cmco) -> (bs,m,co)
        if self.mean: convolved_vals /= nbhd_mask.sum(-1,keepdim=True).clamp(min=1) # Divide by num points
        return convolved_vals

    def forward(self, inp):
        """inputs: [pairs_abq (bs,n,n,d)], [inp_vals (bs,n,ci)]), [query_indices (bs,m)]
           outputs [subsampled_abq (bs,m,m,d)], [convolved_vals (bs,m,co)]"""
        sub_abq, sub_vals, sub_mask, query_indices = self.subsample(inp,withquery=True)
        nbhd_abq, nbhd_vals, nbhd_mask = self.extract_neighborhood(inp, query_indices)
        convolved_vals = self.point_convolve(nbhd_abq, nbhd_vals, nbhd_mask)
        convolved_wzeros = torch.where(sub_mask.unsqueeze(-1),convolved_vals,torch.zeros_like(convolved_vals))
        return sub_abq, convolved_wzeros, sub_mask



@export
def pConvBNrelu(in_channels,out_channels,bn=True,act='swish',**kwargs):
    return nn.Sequential(
        PointConv(in_channels,out_channels,bn=bn,**kwargs),
        MaskBatchNormNd(out_channels) if bn else nn.Sequential(),
        Pass(Swish() if act=='swish' else nn.ReLU(),dim=1)
    )

@export
def LieConvBNrelu(in_channels,out_channels,bn=True,act='swish',**kwargs):
    return nn.Sequential(
        LieConv(in_channels,out_channels,bn=bn,**kwargs),
        MaskBatchNormNd(out_channels) if bn else nn.Sequential(),
        Pass(Swish() if act=='swish' else nn.ReLU(),dim=1)
    )


class BottleBlock(nn.Module):
    """ A bottleneck residual block as described in figure 5"""
    def __init__(self,chin,chout,conv,bn=False,act='swish',fill=None):
        super().__init__()
        assert chin<= chout, f"unsupported channels chin{chin}, chout{chout}. No upsampling atm."
        nonlinearity = Swish if act=='swish' else nn.ReLU
        self.conv = conv(chin//4,chout//4,fill=fill) if fill is not None else conv(chin//4,chout//4)
        self.net = nn.Sequential(
            MaskBatchNormNd(chin) if bn else nn.Sequential(),
            Pass(nonlinearity(),dim=1),
            Pass(nn.Linear(chin,chin//4),dim=1),
            MaskBatchNormNd(chin//4) if bn else nn.Sequential(),
            Pass(nonlinearity(),dim=1),
            self.conv,
            MaskBatchNormNd(chout//4) if bn else nn.Sequential(),
            Pass(nonlinearity(),dim=1),
            Pass(nn.Linear(chout//4,chout),dim=1),
        )
        self.chin = chin
    def forward(self,inp):
        sub_coords, sub_values, mask = self.conv.subsample(inp)
        new_coords, new_values, mask = self.net(inp)
        new_values[...,:self.chin] += sub_values
        return new_coords, new_values, mask

class GlobalPool(nn.Module):
    """computes values reduced over all spatial locations (& group elements) in the mask"""
    def __init__(self,mean=False):
        super().__init__()
        self.mean = mean
        
    def forward(self,x):
        """x [xyz (bs,n,d), vals (bs,n,c), mask (bs,n)]"""
        if len(x)==2: return x[1].mean(1)
        coords, vals,mask = x
        summed = torch.where(mask.unsqueeze(-1),vals,torch.zeros_like(vals)).sum(1)
        if self.mean:
            summed /= mask.sum(-1).unsqueeze(-1)
        return summed

@export
class LieResNet(nn.Module):
    """ Generic LieConv architecture from Fig 5. Relevant Arguments:
        [Fill] specifies the fraction of the input which is included in local neighborhood. 
                (can be array to specify a different value for each layer)
        [nbhd] number of samples to use for Monte Carlo estimation (p)
        [chin] number of input channels: 1 for MNIST, 3 for RGB images, other for non images
        [ds_frac] total downsampling to perform throughout the layers of the net. In (0,1)
        [num_layers] number of BottleNeck Block layers in the network
        [k] channel width for the network. Can be int (same for all) or array to specify individually.
        [liftsamples] number of samples to use in lifting. 1 for all groups with trivial stabilizer. Otherwise 2+
        [Group] Chosen group to be equivariant to.
        [bn] whether or not to use batch normalization. Recommended in all cases except dynamical systems.
        """
    def __init__(self, chin, ds_frac=1, num_outputs=1, k=1536, nbhd=np.inf,
                act="swish", bn=True, num_layers=6, mean=True, per_point=True,pool=True,
                liftsamples=1, fill=1/4, group=SE3,knn=False,cache=False, **kwargs):
        super().__init__()
        if isinstance(fill,(float,int)):
            fill = [fill]*num_layers
        if isinstance(k,int):
            k = [k]*(num_layers+1)
        conv = lambda ki,ko,fill: LieConv(ki, ko, mc_samples=nbhd, ds_frac=ds_frac, bn=bn,act=act, mean=mean,
                                group=group,fill=fill,cache=cache,knn=knn,**kwargs)
        self.net = nn.Sequential(
            Pass(nn.Linear(chin,k[0]),dim=1), #embedding layer
            *[BottleBlock(k[i],k[i+1],conv,bn=bn,act=act,fill=fill[i]) for i in range(num_layers)],
            MaskBatchNormNd(k[-1]) if bn else nn.Sequential(),
            Pass(Swish() if act=='swish' else nn.ReLU(),dim=1),
            Pass(nn.Linear(k[-1],num_outputs),dim=1),
            GlobalPool(mean=mean) if pool else Expression(lambda x: x[1]),
            )
        self.liftsamples = liftsamples
        self.per_point=per_point
        self.group = group

    def forward(self, x):
        lifted_x = self.group.lift(x,self.liftsamples)
        return self.net(lifted_x)

@export
class ImgLieResnet(LieResNet):
    """ Lie Conv architecture specialized to images. Uses scaling rule to determine channel
         and downsampling scaling. Same arguments as LieResNet"""
    def __init__(self,chin=1,total_ds=1/64,num_layers=6,group=T(2),fill=1/32,k=256,
        knn=False,nbhd=12,num_targets=10,increase_channels=True,**kwargs):
        ds_frac = (total_ds)**(1/num_layers)
        fill = [fill/ds_frac**i for i in range(num_layers)]
        if increase_channels: # whether or not to scale the channels as image is downsampled
            k = [int(k/ds_frac**(i/2)) for i in range(num_layers+1)]
        super().__init__(chin=chin,ds_frac=ds_frac,num_layers=num_layers,nbhd=nbhd,mean=True,
                        group=group,fill=fill,k=k,num_outputs=num_targets,cache=True,knn=knn,**kwargs)
        self.lifted_coords = None

    def forward(self,x,coord_transform=None):
        """ assumes x is a regular image: (bs,c,h,w)"""
        bs,c,h,w = x.shape
        # Construct coordinate grid
        i = torch.linspace(-h/2.,h/2.,h)
        j = torch.linspace(-w/2.,w/2.,w)
        coords = torch.stack(torch.meshgrid([i,j]),dim=-1).float()
        # Perform center crop
        center_mask = coords.norm(dim=-1)<15. # crop out corners (filled only with zeros)
        coords = coords[center_mask].view(-1,2).unsqueeze(0).repeat(bs,1,1).to(x.device)
        if coord_transform is not None: coords = coord_transform(coords)
        values = x.permute(0,2,3,1)[:,center_mask,:].reshape(bs,-1,c)
        mask = torch.ones(bs,values.shape[1],device=x.device)>0 # all true
        z = (coords,values,mask)
        # Perform lifting of the coordinates and cache results
        with torch.no_grad():
            if self.lifted_coords is None:
                self.lifted_coords,lifted_vals,lifted_mask = self.group.lift(z,self.liftsamples)
            else:
                lifted_vals,lifted_mask = self.group.expand_like(values,mask,self.lifted_coords)
        return self.net((self.lifted_coords,lifted_vals,lifted_mask))






