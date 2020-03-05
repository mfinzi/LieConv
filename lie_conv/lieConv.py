

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
    """assumes that the inputs to the net are shape (bs,n,nbhd,c)"""
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
    def __init__(self,chin,chout,nbhd=32,xyz_dim=3,ds_frac=1,knn_channels=None,act='swish',bn=False,mean=False):
        super().__init__()
        self.chin = chin # input channels
        self.cmco_ci = 16 # a hyperparameter controlling size and bottleneck compute cost of weightnet
        self.xyz_dim = xyz_dim # dimension of the space on which convolution operates
        self.knn_channels = knn_channels # number of xyz dims on which to compute knn
        self.nbhd = nbhd # neighborhood (support of the convolution filter) size
        self.weightnet = WeightNet(xyz_dim, self.cmco_ci, act, bn)
        self.linear = nn.Linear(self.cmco_ci * chin, chout)
        self.mean=mean
        assert ds_frac==1, "test branch no ds, will need to carefully check that subsample respects padding"
        self.subsample = FarthestSubsample(ds_frac,knn_channels=knn_channels)

    def extract_neighborhood(self,inp,query_xyz):
        """ inputs shape ([inp_xyz (bs,n,d)], [inp_vals (bs,n,c)], [query_xyz (bs,m,d)])"""
        inp_xyz,inp_vals,mask = inp
        neighbor_idx = knn_point(min(self.nbhd, inp_xyz.shape[1]),
                    inp_xyz[:,:,:self.knn_channels], query_xyz[:,:,:self.knn_channels],mask)#self.neighbor_lookup[key]\
        neighbor_xyz = index_points(inp_xyz, neighbor_idx) # (bs,n,nbhd,d)
        neighbor_values = index_points(inp_vals, neighbor_idx) #(bs,n,nbhd,c)
        neighbor_mask = index_points(mask,neighbor_idx) # (bs,n,nbhd)
        return neighbor_xyz, neighbor_values, neighbor_mask

    def point_convolve(self,embedded_group_elems,nbhd_vals, nbhd_mask):
        """ embedded_group_elems: (bs,m,nbhd,d)
            nbhd_vals: (bs,m,nbhd,ci)
            nbhd_mask: (bs,m,nbhd)"""
        bs, m, nbhd, ci = nbhd_vals.shape  # (bs,m,nbhd,d) -> (bs,m,nbhd,cm*co/ci)
        _, penult_kernel_weights, _ = self.weightnet((None,embedded_group_elems,nbhd_mask))
        penult_kernel_weights_m = torch.where(nbhd_mask.unsqueeze(-1),penult_kernel_weights,torch.zeros_like(penult_kernel_weights))
        nbhd_vals_m = torch.where(nbhd_mask.unsqueeze(-1),nbhd_vals,torch.zeros_like(nbhd_vals))
        #      (bs,m,nbhd,ci) -> (bs,m,ci,nbhd) @ (bs, m, nbhd, cmco/ci) -> (bs,m,ci,cmco/ci) -> (bs,m, cmco) 
        partial_convolved_vals = (nbhd_vals_m.transpose(-1,-2)@penult_kernel_weights_m).view(bs, m, -1)
        convolved_vals = self.linear(partial_convolved_vals)#/np.sqrt(ci) #  (bs,m,cmco) -> (bs,m,co)
        if self.mean: convolved_vals /= nbhd_mask.sum(-1,keepdim=True).clamp(min=1) # why so Nan
        return convolved_vals

    def get_embedded_group_elems(self,output_xyz,nbhd_xyz):
        return output_xyz - nbhd_xyz
    
    def forward(self, inp):
        """inputs, and outputs have shape ([xyz (bs,n,d)], [vals (bs,n,c)])
            query_xyz has shape (bs,n,d)"""
        #xyz (bs,n,c), new_xyz (bs,m,c) neighbor_xyz (bs,m,nbhd,c)
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
    #print(mask[B,farthest])
    for i in range(m):
        chosen_indices[:, i] = farthest # add point that is farthest to chosen
        dist = torch.where(mask,dists[B,farthest],-100*torch.ones_like(distances)) # (bs,n) compute distance from new point to all others
        closer = dist < distances      # if dist from new point is smaller than chosen points so far
        distances[closer] = dist[closer] # update the chosen set's distance to all other points
        farthest = torch.max(distances, -1)[1] # select the point that is farthest from the set
    #good_ids = mask[B[:,None].expand(*chosen_indices.shape),chosen_indices]
    #print(torch.all(good_ids),good_ids.sum(),bs,m)
    return chosen_indices


class FPSsubsample(nn.Module):
    def __init__(self,ds_frac,cache=False,group=None):
        super().__init__()
        self.ds_frac = ds_frac
        self.cache=cache
        self.cached_indices = None
        self.group = group
    def forward(self,inp,withquery=False):
        ab_pairs,vals,mask = inp
        dist = self.group.distance if self.group else lambda ab: norm(ab,dim=-1)
        if self.ds_frac!=1:
            if self.cache and self.cached_indices is None:
                query_idx = self.cached_indices = FPSindices(dist(ab_pairs),self.ds_frac,mask).detach()
            elif self.cache:
                query_idx = self.cached_indices
            else:
                query_idx = FPSindices(dist(ab_pairs),self.ds_frac,mask)
            B = torch.arange(query_idx.shape[0],device=query_idx.device).long()[:,None]
            subsampled_ab_pairs = ab_pairs[B,query_idx][B,:,query_idx]
            subsampled_values = vals[B,query_idx]
            subsampled_mask = mask[B,query_idx]
        else:
            subsampled_ab_pairs = ab_pairs
            subsampled_values = vals
            subsampled_mask = mask
            query_idx = None
        if withquery: return (subsampled_ab_pairs,subsampled_values,subsampled_mask, query_idx)
        return (subsampled_ab_pairs,subsampled_values,subsampled_mask)

def sample_within_ball(within_ball,k,bs):
    """ inputs: [within_ball (bs,m,n)] int k
        outputs: [indices (bs,m,k)]"""
    m,n = within_ball.shape[1:]
    with torch.no_grad():
        max_k = torch.max(within_ball.sum(dim=-1))
    k_to_use = min(max_k,k)
    fixed_coordinates = (bs!=within_ball.shape[0])
    if fixed_coordinates:
        
        mask_restricted, idx_restricted =  torch.topk(within_ball.float(), k=max_k, dim=-1,largest=True,sorted=False)
        # (1,m,k) (1,m,k)
        noise = torch.zeros_like(mask_restricted).repeat((bs,1,1))
        noise.uniform_(0,1)
        mask_restricted_subsampled, idx_restricted_subsampled = torch.topk(mask_restricted+noise, k=k_to_use, dim=-1,largest=True,sorted=False)
        # (bs,m,k)
        M = torch.arange(m).long().to(idx_restricted.device)[None,:,None]
        idx_subsampled = idx_restricted[0,M,idx_restricted_subsampled]
        return idx_subsampled
    else:
        noise = torch.zeros(bs,m,n,device=within_ball.device)
        noise.uniform_(0,1)
        valid_within_ball, nbhd_idx =torch.topk(within_ball+noise,k_to_use,dim=-1,largest=True,sorted=False)
        valid_within_ball = (valid_within_ball>1)
        return nbhd_idx


class LieConv(PointConv):
    def __init__(self,*args,group=SE3,ds_frac=1,fill=1/3,cache=False,knn=False,**kwargs):
        kwargs.pop('xyz_dim',None)
        self.group = group
        self.r = 2#radius
        self.fill_frac = min(fill,1.)
        self.knn=knn
        super().__init__(*args,xyz_dim=group.embed_dim+2*group.q_dim,**kwargs)
        self.subsample = FPSsubsample(ds_frac,cache=cache,group=self.group)
        self.coeff = .5
        self.fill_frac_ema = fill
    

    def extract_neighborhood(self,inp,query_indices):
        """ inputs: [pairs_ab (bs,n,n,d), inp_vals (bs,n,c), mask (bs,n), query_indices (bs,m)]
            outputs: [neighbor_ab (bs,m,nbhd,d), neighbor_vals (bs,m,nbhd,c), nbhd_mask (bs,m, nbhd)]"""
        
        pairs_ab, inp_vals, mask = inp
        if query_indices is not None:
            B = torch.arange(pairs_ab.shape[0],device=inp_vals.device).long()[:,None]
            ab_at_query = pairs_ab[B,query_indices]
            mask_at_query = mask[B,query_indices]
        else:
            ab_at_query = pairs_ab
            mask_at_query = mask
        vals_at_query = inp_vals
        dists = self.group.distance(ab_at_query) #(bs,m,n,d) -> (bs,m,n)
        dists = torch.where(mask[:,None,:].expand(*dists.shape),dists,1e8*torch.ones_like(dists))
        k = min(self.nbhd,inp_vals.shape[1])
        # NBHD: Subsampling within the ball
        _,m,n = dists.shape
        bs = inp_vals.shape[0]
        if self.knn: # NBHD: KNN
            nbhd_idx = torch.topk(dists,k,dim=-1,largest=False,sorted=False)[1] #(bs,m,nbhd)
            valid_within_ball = (nbhd_idx>-1)&mask[:,None,:]&mask_at_query[:,:,None]
            assert not torch.any(nbhd_idx>dists.shape[-1]), f"error with topk,\
                        nbhd{k} nans|inf{torch.any(torch.isnan(dists)|torch.isinf(dists))}"
        else: # NBHD: Sampled Distance Ball
            valid_within_ball = (dists < self.r)&mask[:,None,:]&mask_at_query[:,:,None] # (bs,m,n)
            nbhd_idx = sample_within_ball(valid_within_ball,k,bs)

        B = torch.arange(bs,device=inp_vals.device).long()[:,None,None].expand(*nbhd_idx.shape)
        b = B if dists.shape[0]==bs else 0
        M = torch.arange(ab_at_query.shape[1],device=inp_vals.device).long()[None,:,None].expand(*nbhd_idx.shape)
        nbhd_ab = ab_at_query[b,M,nbhd_idx]   #(bs,m,n,d) -> (bs,m,nbhd,d)
        nbhd_vals = vals_at_query[B,nbhd_idx] #(bs,n,c)   -> (bs,m,nbhd,c)
        nbhd_mask = mask[b,nbhd_idx]          #(bs,n)     -> (bs,m,nbhd)
        #print(nbhd_ab.shape,nbhd_vals.shape,nbhd_mask.shape)
        navg = (valid_within_ball.float()).sum(-1).sum()/mask_at_query[:,:,None].sum()
        if self.training:
            avg_fill = (navg/mask.sum(-1).float().mean()).cpu().item()
            self.r +=  self.coeff*(self.fill_frac - avg_fill)#self.fill_frac*n/navg.cpu().item()-1)
            self.fill_frac_ema += .1*(avg_fill-self.fill_frac_ema)
            #print(avg_fill)
        return nbhd_ab, nbhd_vals, (nbhd_mask&valid_within_ball[b,M,nbhd_idx].bool())#W
    # def log_data(self,logger,step,name):
    #     logger.add_scalars('info', {f'{name}_fill':self.fill_frac_ema}, step=step)
    #     logger.add_scalars('info', {f'{name}_R':self.r}, step=step)

    def point_convolve(self,embedded_group_elems,nbhd_vals,nbhd_mask):
        """ embedded_group_elems: (bs,m,nbhd,d)
            nbhd_vals: (bs,m,nbhd,ci)
            nbhd_mask: (bs,m,nbhd) """
        bs, m, nbhd, ci = nbhd_vals.shape  # (bs,m,nbhd,d) -> (bs,m,nbhd,cm*co/ci)
        _, penult_kernel_weights, _ = self.weightnet((None,embedded_group_elems,nbhd_mask))
        penult_kernel_weights_m = torch.where(nbhd_mask.unsqueeze(-1),penult_kernel_weights,torch.zeros_like(penult_kernel_weights))
        nbhd_vals_m = torch.where(nbhd_mask.unsqueeze(-1),nbhd_vals,torch.zeros_like(nbhd_vals))
        #      (bs,m,nbhd,ci) -> (bs,m,ci,nbhd) @ (bs, m, nbhd, cmco/ci) -> (bs,m,ci,cmco/ci) -> (bs,m, cmco) 
        partial_convolved_vals = (nbhd_vals_m.transpose(-1,-2)@penult_kernel_weights_m).view(bs, m, -1)
        convolved_vals = self.linear(partial_convolved_vals) #  (bs,m,cmco) -> (bs,m,co)
        if self.mean: convolved_vals /= nbhd_mask.sum(-1,keepdim=True).clamp(min=1)
        return convolved_vals

    def forward(self, inp):
        """inputs: ([pairs_ab (bs,n,n,d)], [inp_vals (bs,n,ci)]), [query_indices (bs,m)]
           outputs ([subbed_ab (bs,m,m,d)], [convolved_vals (bs,m,co)])"""
        sub_ab, sub_vals, sub_mask, query_indices = self.subsample(inp,withquery=True)
        nbhd_ab, nbhd_vals, nbhd_mask = self.extract_neighborhood(inp, query_indices)
        convolved_vals = self.point_convolve(nbhd_ab, nbhd_vals, nbhd_mask)
        convolved_wzeros = torch.where(sub_mask.unsqueeze(-1),convolved_vals,torch.zeros_like(convolved_vals))
        # assert not torch.any(torch.isnan(convolved_wzeros)|torch.isinf(convolved_wzeros)), f"nans|inf{torch.any(torch.isnan(convolved_wzeros)|torch.isinf(convolved_wzeros))}"
        return sub_ab, convolved_wzeros, sub_mask



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
    def __init__(self,chin,chout,conv,bn=False,act='swish',fill=None):
        super().__init__()
        assert chin<= chout, f"unsupported channels chin{chin}, chout{chout}"
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
    """computes values reduced over all spatial locations in the mask"""
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
class LieResNet(nn.Module,metaclass=Named):
    def __init__(self, chin, ds_frac=1, num_outputs=1, k=1536, nbhd=np.inf,
                act="swish", bn=True, num_layers=6, mean=True, per_point=True,pool=True,
                liftsamples=1, fill=1/32, group=SE3,knn=False,cache=False, **kwargs):
        super().__init__()
        if isinstance(fill,(float,int)):
            fill = [fill]*num_layers
        if isinstance(k,int):
            k = [k]*(num_layers+1)
        conv = lambda ki,ko,fill: LieConv(ki, ko, nbhd=nbhd, ds_frac=ds_frac, bn=bn,act=act, mean=mean,
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
    def __init__(self,chin=1,total_ds=1/64,num_layers=6,group=T(2),fill=1/32,k=256,
        knn=False,nbhd=12,num_targets=10,increase_channels=True,**kwargs):
        ds_frac = (total_ds)**(1/num_layers) if num_layers else 1
        fill = [fill/ds_frac**i for i in range(num_layers)]
        if increase_channels:
            k = [int(k/ds_frac**(i/2)) for i in range(num_layers+1)]
        super().__init__(chin=chin,ds_frac=ds_frac,num_layers=num_layers,nbhd=nbhd,mean=True,
                        group=group,fill=fill,k=k,num_outputs=num_targets,cache=True,knn=knn,**kwargs)
        self.lifted_coords = None
    def forward(self,x,coord_transform=None):
        """ assumes x is a regular image: (bs,c,h,w)"""
        bs,c,h,w = x.shape
        i = torch.linspace(-h/2.,h/2.,h)
        j = torch.linspace(-w/2.,w/2.,w)
        coords = torch.stack(torch.meshgrid([i,j]),dim=-1).float()
        center_mask = coords.norm(dim=-1)<15. # crop out corners (filled only with zeros)
        coords = coords[center_mask].view(-1,2).unsqueeze(0).repeat(bs,1,1).to(x.device)
        if coord_transform is not None: coords = coord_transform(coords)
        values = x.permute(0,2,3,1)[:,center_mask,:].reshape(bs,-1,c)
        cache=True
        mask = torch.ones(1 if cache else bs,values.shape[1],device=x.device)>0 # all true # if test_lift else bs
        z = (coords,values,mask)
        with torch.no_grad():
            z_bs1 = (coords[:1],values[:1],mask[:1]) if cache else z
            if self.lifted_coords is None:
                self.lifted_coords,_,_ = self.group.lift(z_bs1,self.liftsamples)
            lifted_vals,lifted_mask = self.group.expand_like(values,mask,self.lifted_coords)
            # if self.lifted_coords is None:
            #     self.lifted_coords,lifted_vals,lifted_mask = self.group.lift(z,self.liftsamples)
            # else:
            #     lifted_vals,lifted_mask = self.group.expand_like(values,mask,self.lifted_coords)
            # if test_lift:
            #     z_bs1 = (coords[:1],values[:1],mask[:1])
            #     self.lifted_coords,_,_ = self.group.lift(z_bs1,self.liftsamples)
            #     lifted_vals,lifted_mask = self.group.expand_like(values,mask,self.lifted_coords)
            # else:
            #     self.lifted_coords,_,_ = self.group.lift(z,self.liftsamples)
            #     lifted_vals,lifted_mask = self.group.expand_like(values,mask,self.lifted_coords)
        return self.net((self.lifted_coords,lifted_vals,lifted_mask))






