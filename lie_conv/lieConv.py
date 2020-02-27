"""
Utility functions adapted from https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/utils.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lie_conv.utils import Expression,export,Named
from lie_conv.lieGroups import T,SO2,SO3,SE2,SE3, norm
from lie_conv.masked_batchnorm import MaskBatchNormNd


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    if src.shape[1]==1 or dst.shape[1]==1:
        return torch.sum((src - dst) ** 2, -1)
    #print(src.shape,dst.shape)
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, ...]
    return new_points

def farthest_point_sample(xyz, npoint, distance = square_distance):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    #import ipdb; ipdb.set_trace()
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distances = torch.ones(B, N).to(device) * 1e8
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = distance(xyz,centroid)#torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distances #TODO check that broadcasting of above is correct 
        distances[mask] = dist[mask]
        farthest = torch.max(distances, -1)[1]
    return centroids

def farthest_ball_point(radius,nsample,xyz,new_xyz,distance=square_distance):
    # two things to fix: 
    # 1) random -> farthest or low discrepancy sampling of elements in the ball
    # 2) when less than nsample elements in ball, don't fill with first elem
    #       rather: also give a mask to use in point convolve
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def knn_point(nbhd, xyz, new_xyz, mask, distance=square_distance):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [bs, n, c]
        new_xyz: query points, [bs, m, c]
        mask: the valid mask [bs, n]
    Return:
        group_idx: grouped points index, [bs, m, nbhd]
    """
    
    sqrdists = distance(new_xyz.unsqueeze(-2), xyz.unsqueeze(-3)) #(bs,m,n)
    #print('sqrdists',sqrdists.shape)
    sqrdists[~mask[:,None,:].expand(*sqrdists.shape)] = 1e8 # topk doesn't like infs or nans
    _, group_idx = torch.topk(sqrdists,nbhd, dim = -1, largest=False, sorted=False)
    if torch.any(group_idx>10000): # This means there was an error, nans in input?
        print("greater than 10k :(")
        print(xyz.shape)
        print(new_xyz.shape)
        print(xyz[0])
        print(new_xyz[0])
        raise Exception
    return group_idx

def pthash(xyz):
    """ a hash function for pytorch arrays """
    return hash(tuple(xyz.cpu().data.numpy().reshape(-1))+(xyz.device,))

@export
class FarthestSubsample(nn.Module):
    def __init__(self,ds_frac=0.5,knn_channels=None,distance=square_distance,cache=False):
        super().__init__()
        self.ds_frac = ds_frac
        self.subsample_lookup = {}
        self.knn_channels = knn_channels
        self.distance=distance
        self.cache=cache
    def forward(self,x,coords_only=False):
        """ inputs shape ([inp_xyz (bs,n,d)], [inp_vals (bs,n,c)]"""
        coords,values,mask = x
        if self.ds_frac==1:
            if coords_only: return coords
            else: return x
        num_downsampled_points = int(np.round(coords.shape[1]*self.ds_frac))
        fps_idx = farthest_point_sample(coords[:,:,:self.knn_channels],
                            num_downsampled_points,distance=self.distance)#self.subsample_lookup[key]
        new_coords = index_points(coords,fps_idx)
        if coords_only: return new_coords
        new_values = index_points(values,fps_idx)
        new_mask = index_points(mask,fps_idx)
        return new_coords,new_values,new_mask

def compute_density(xyz, bandwidth):
    '''
    xyz: input points position data, [B, N, C]
    '''
    #import ipdb; ipdb.set_trace()
    B, N, C = xyz.shape
    sqrdists = square_distance(xyz, xyz)
    gaussion_density = torch.exp(- sqrdists / (2.0 * bandwidth * bandwidth)) / (2.5 * bandwidth)
    xyz_density = gaussion_density.mean(dim = -1)

    return xyz_density

class GlobalPool(nn.Module):
    """computes values reduced over all spatial locations excluding nans"""
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
        #print(torch.isnan(summed).sum())
        #print(mask.sum(-1)[0])
        return summed

@export
def Swish():
    return Expression(lambda x: x*F.sigmoid(x))

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

# def denan(x):
#     """ replace nans in x with 0s"""
#     return torch.where(torch.isnan(x),torch.zeros_like(x),x)

# def renan(x,y):
#     """ introduce nans in x where they occur in y (assuming the same for each ch),
#         also divide by the number of non nans if mean=True"""
#     nans_like_x = torch.isnan(y[...,:1]).expand(*x.shape)
#     #print(f"numnan: {torch.isnan(y[0,...,0]).sum()}")
#     #print(f"unintentional xnans {torch.isnan(x[0,...,0]).sum()}")
#     #print(f"sums: {sums[0,:,0]}")
#     x_with_nans = torch.where(nans_like_x,torch.zeros_like(x)/0,x)
#     #print(f"sums after: {(~torch.isnan(x_with_nans[0,...,0])).sum()}")
#     #if mean: x_with_nans = x_with_nans / 
#     return x_with_nans

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
        convolved_vals = self.linear(partial_convolved_vals) #  (bs,m,cmco) -> (bs,m,co)
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


# class GroupPointConv(PointConvBase):
#     def __init__(self,*args,group=SE2,**kwargs):
#         kwargs.pop('xyz_dim')
#         self.group = group
#         super().__init__(*args,xyz_dim=group.embed_dim,**kwargs)
        
#     def extract_neighborhood(self,inp_a,inp_vals,query_a):
#         neighbor_idx = knn_point(min(self.nbhd,inp_a.shape[1]),inp_a[:,:,:self.knn_channels],
#                     query_a[:,:,:self.knn_channels],self.group.distance)#self.neighbor_lookup[key]
#         #print(inp_a.shape)
#         #print(neighbor_idx.shape)
#         nidx = neighbor_idx.cpu().data.numpy()
#         neighbor_a = index_points(inp_a, neighbor_idx) # [B, npoint, nsample, C]
#         nidx2 = neighbor_idx.cpu().data.numpy()
#         neighbor_values = index_points(inp_vals, neighbor_idx)
#         #print('neighbor_a',neighbor_a.shape)
#         return neighbor_a, neighbor_values # (bs,n,nbhd,c)
#     def get_embedded_group_elems(self,output_a,nbhd_a):
#         #print(output_a.shape)
#         #print(nbhd_a.shape)
#         return self.group.BCH(output_a,-nbhd_a)


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
            outputs: [neighbor_ab (bs,m,nbhd,d), neighbor_vals (bs,m,nbhd,c)]"""
        
        pairs_ab, inp_vals, mask = inp
        #print(inp_vals[mask].sum())
        if query_indices is not None:
            B = torch.arange(inp_vals.shape[0],device=inp_vals.device).long()[:,None]
            ab_at_query = pairs_ab[B,query_indices]
            mask_at_query = mask[B,query_indices]
        else:
            ab_at_query = pairs_ab
            mask_at_query = mask
        vals_at_query = inp_vals
        #print(vals_at_query[mask].sum())
        dists = self.group.distance(ab_at_query) #(bs,m,n,d) -> (bs,m,n)
        dists = torch.where(mask[:,None,:].expand(*dists.shape),dists,1e8*torch.ones_like(dists))
        #dists[~mask[:,None,:].expand(*dists.shape)] = 1e8
        #print((dists<1e7).float().sum()/(mask_at_query[:,None,:].expand(*dists.shape).sum()))
        #print(dists[mask_at_query].sum())
        #print(dists[mask_at_query].sum())
        k = min(self.nbhd,inp_vals.shape[1])
        # NBHD: Subsampling within the ball
        bs,m,n = dists.shape
        #print(dists.shape)
        if self.knn:
            nbhd_idx = torch.topk(dists,k,dim=-1,largest=False,sorted=False)[1] #(bs,m,nbhd)
            valid_within_ball = (nbhd_idx>-1)&mask[:,None,:]&mask_at_query[:,:,None]
            assert not torch.any(nbhd_idx>dists.shape[-1]), f"error with topk,\
                        nbhd{k} nans|inf{torch.any(torch.isnan(dists)|torch.isinf(dists))}"
        else:
            #print(((mask[:,None,:]&mask[:,:,None]).sum().float()/(bs)).sqrt())
            within_ball = (dists < self.r)&mask[:,None,:]&mask_at_query[:,:,None] # (bs,m,n)
            #navg = within_ball.sum().float()/mask_at_query.sum().float()

            #print(f'average_number_within_ball({self.r}) of max {n}, {navg}')
            #print((dists*(mask[:,None,:]&mask[:,:,None]).float()).sum()/(mask[:,None,:]&mask[:,:,None]).float().sum())
            #print(within_ball.sum().float()/mask[:,:,None].sum())#(mask[:,None,:]&mask[:,:,None]).float().sum())
            #rand_perm = torch.rand(bs,m,n).argsort(dim=-1) # (bs,m,n)
            #rand_perm = torch.cat([torch.randperm(n)[None, :] for _ in range(bs * m)], dim=0).reshape(bs,m,n)
            #inverse_perm_indices = rand_perm.argsort(dim=-1) # (bs,m,n)
            B = torch.arange(bs)[:,None,None]#.expand(*random_perm.shape)
            M = torch.arange(m)[None,:,None]#.expand(*random_perm.shape)
            #permed_within_ball = within_ball[B.expand(*rand_perm.shape),M.expand(*rand_perm.shape),rand_perm]
            noise = torch.zeros(bs,m,n,device=within_ball.device)
            noise.uniform_(0,1)
            valid_within_ball, nbhd_idx =torch.topk(within_ball+noise,k,dim=-1,largest=True,sorted=False)
            valid_within_ball = (valid_within_ball>1)
            # #nbhd_idx = rand_perm[B.expand(*permed_nbhd_idx.shape),M.expand(*permed_nbhd_idx.shape),permed_nbhd_idx]
            # nbhd_idx = permed_nbhd_idx
            # #print(valid_within_ball)
            # #randidx = torch.randint(0, n, (bs,m,k), dtype=torch.long) #choose random start
            # #inverse_permutation_indexes = torch.empty_like(permutation_indexes).scatter_(dim=1, index=permutation_indexes, src=y)
            # #B = torch.arange(bs, dtype=torch.long).to(device)
        # NBHD: KNN
        #nbhd_idx = torch.topk(dists,k,dim=-1,largest=False,sorted=False)[1] #(bs,m,nbhd)

        #print(nbhd_idx[mask_at_query].sum())
        #print("nbhd idx", nbhd_idx.shape)
        
        B = torch.arange(inp_vals.shape[0],device=inp_vals.device).long()[:,None,None].expand(*nbhd_idx.shape)
        M = torch.arange(ab_at_query.shape[1],device=inp_vals.device).long()[None,:,None].expand(*nbhd_idx.shape)
        nbhd_ab = ab_at_query[B,M,nbhd_idx]  #(bs,m,n,d) -> (bs,m,nbhd,d)
        nbhd_vals = vals_at_query[B,nbhd_idx]#(bs,n,c) -> (bs,m,nbhd,c)
        nbhd_mask = mask[B,nbhd_idx]         #(bs,n) -> (bs,m,nbhd)
        #print("mask mean",mask.float().sum(-1).mean())
        #print("average number of valid neighborhood points per real atom",
        #(nbhd_mask*valid_within_ball.float()).sum(-1).sum()/mask_at_query[:,:,None].sum())
        navg = (within_ball.float()).sum(-1).sum()/mask_at_query[:,:,None].sum()
        if self.training:
            avg_fill = (navg/mask.sum(-1).float().mean()).cpu().item()
            self.r +=  self.coeff*(self.fill_frac - avg_fill)#self.fill_frac*n/navg.cpu().item()-1)
            self.fill_frac_ema += .1*(avg_fill-self.fill_frac_ema)
        #print(nbhd_vals.shape)
        #print(nbhd_vals[nbhd_mask].sum())
        #assert False
        return nbhd_ab, nbhd_vals, nbhd_mask#(nbhd_mask&valid_within_ball.bool())
    # def log_data(self,logger,step,name):
    #     logger.add_scalars('info', {f'{name}_fill':self.fill_frac_ema}, step=step)
    #     logger.add_scalars('info', {f'{name}_R':self.r}, step=step)

    def point_convolve(self,embedded_group_elems,nbhd_vals,nbhd_mask):
        """ embedded_group_elems: (bs,m,nbhd,d)
            nbhd_vals: (bs,m,nbhd,ci)
            nbhd_mask: (bs,m,nbhd)
            """
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
        return sub_ab, convolved_wzeros, sub_mask

def FPSindices(dists,frac,mask):
    """ inputs: pairwise distances DISTS (bs,n,n), downsample_frac (float), valid atom mask (bs,n) 
        outputs: chosen_indices (bs,m) """
    m = int(np.round(frac*dists.shape[1]))
    device = dists.device
    bs,n = dists.shape[:2]
    chosen_indices = torch.zeros(bs, m, dtype=torch.long,device=device)
    distances = torch.ones(bs, n,device=device) * 1e8
    #m1,m2 = torch.where(~mask)
    #distances = torch.where(mask,distances,-1*torch.ones_like(distances)).transpose(0,1)
    #distances = torch.where(mask,distances,-1*torch.ones_like(distances)).transpose(0,1)
    #print(torch.where(mask))
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
            Pass(nn.Linear(k[-1],k[-1]//2),dim=1),
            Pass(Swish() if act=='swish' else nn.ReLU(),dim=1),
            #Pass(nn.Linear(k[-1]//2,num_outputs),dim=1),
            GlobalPool(mean=mean) if pool else Expression(lambda x: x[1]),
            nn.Linear(k[-1]//2,num_outputs)
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
        ds_frac = (total_ds)**(1/num_layers)
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
        mask = torch.ones(bs,values.shape[1],device=x.device)>0 # all true
        z = (coords,values,mask)
        with torch.no_grad():
            if self.lifted_coords is None:
                self.lifted_coords,lifted_vals,lifted_mask = self.group.lift(z,self.liftsamples)
            else:
                lifted_vals,lifted_mask = self.group.expand_like(values,mask,self.lifted_coords)
        return self.net((self.lifted_coords,lifted_vals,lifted_mask))
# @export
# class both(nn.Module):
#     def __init__(self,module1,module2):
#         super().__init__()
#         self.module1 = module1
#         self.module2 = module2
#     def forward(self,inp):
#         x,z = inp
#         return self.module1(x),self.module2(z)

@export
class Pass(nn.Module):
    def __init__(self,module,dim=1):
        super().__init__()
        self.module = module
        self.dim=dim
    def forward(self,x):
        xs = list(x)
        xs[self.dim] = self.module(xs[self.dim])
        return xs


# @export
# class pBottleneck(nn.Module):
#     def __init__(self,in_channels,out_channels,nbhd=3.66**2,ds_frac=0.5,drop_rate=0,r=4,gn=False,**kwargs):
#         super().__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         norm_layer = nn.BatchNorm1d
#         self.pointconv = PointConv(in_channels//r,out_channels,nbhd=nbhd,ds_frac=ds_frac,**kwargs)
#         self.net = nn.Sequential(
#             Pass(norm_layer(in_channels)),
#             Pass(Swish() if act=='swish' else nn.ReLU()),
#             Pass(nn.Conv1d(in_channels,in_channels//r,1)),
#             Pass(norm_layer(in_channels//r)),
#             Pass(Swish() if act=='swish' else nn.ReLU()),
#             self.pointconv,
#             #Pass(norm_layer(out_channels)),
#             #Pass(nn.ReLU()),
#             #Pass(nn.Conv1d(out_channels,out_channels,1)),
#         )

#     def forward(self,x):
#         #coords,values = x
#         #print(values.shape)
#         new_coords,new_values  = self.net(x)
#         new_values[:,:self.in_channels] += self.pointconv.subsample(x)[1] # subsampled old values
#         #print(shortcut.shape)
#         #print(new_coords.shape,new_values.shape)
#         return new_coords,new_values



# def imagelike_nn_downsample(x,coords_only=False):
#     coords,values = x
#     bs,c,N = values.shape
#     h = w = int(np.sqrt(N))
#     ds_coords = torch.nn.functional.interpolate(coords.view(bs,2,h,w),scale_factor=0.5)
#     ds_values = torch.nn.functional.interpolate(values.view(bs,c,h,w),scale_factor=0.5)
#     if coords_only: return ds_coords.view(bs,2,-1)
#     return ds_coords.view(bs,2,-1), ds_values.view(bs,c,-1)

# def concat_coords(x):
#     bs,c,h,w = x.shape
#     coords = torch.stack(torch.meshgrid([torch.linspace(-1,1,h),torch.linspace(-1,1,w)]),dim=-1).view(h*w,2).unsqueeze(0).permute(0,2,1).repeat(bs,1,1).to(x.device)
#     inp_as_points = x.view(bs,c,h*w)
#     return (coords,inp_as_points)
# def uncat_coords(x):
#     bs,c,n = x[1].shape
#     h = w = int(np.sqrt(n))
#     return x[1].view(bs,c,h,w)

# @export
# def imgpConvBNrelu(in_channels,out_channels,**kwargs):
#     return nn.Sequential(
#         Expression(concat_coords),
#         PointConv(in_channels,out_channels,**kwargs),
#         Expression(uncat_coords),
#         nn.BatchNorm2d(out_channels),
#         nn.ReLU(),
#     )

# @export
# class pResBlock(nn.Module):
#     def __init__(self,in_channels,out_channels,drop_rate=0,stride=1,nbhd=3**2,xyz_dim=2):
#         super().__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         if xyz_dim==2:
#             ds = 1 if stride==1 else imagelike_nn_downsample#
#         elif xyz_dim==3:
#             ds = 1 if stride==1 else .25
            
#         self.net = nn.Sequential(
#             Pass(nn.BatchNorm1d(in_channels)),
#             Pass(nn.ReLU()),
#             PointConv(in_channels,out_channels,nbhd=nbhd,
#             ds_frac=1 if xyz_dim==2 else np.sqrt(ds),xyz_dim=xyz_dim),
#             Pass(nn.Dropout(p=drop_rate)),
#             Pass(nn.BatchNorm1d(out_channels)),
#             Pass(nn.ReLU()),
#             PointConv(out_channels,out_channels,nbhd=nbhd,
#             ds_frac=ds if xyz_dim==2 else np.sqrt(ds),xyz_dim=xyz_dim),
#         )
#         self.shortcut = nn.Sequential()
#         if in_channels!=out_channels:
#             self.shortcut.add_module('conv',Pass(nn.Conv1d(in_channels,out_channels,1)))
#         if stride!=1:
#             if xyz_dim==2:
#                 self.shortcut.add_module('ds',Expression(lambda a: imagelike_nn_downsample(a)))
#             elif xyz_dim==3:
#                 self.shortcut.add_module('ds',FarthestSubsample(ds_frac=ds))

#     def forward(self,x):
#         res_coords,res_values = self.net(x)
#         skip_coords,skip_values = self.shortcut(x)
#         return res_coords,res_values+skip_values






# @export
# class LearnedSubsample(nn.Module):
#     def __init__(self,ds_frac=0.5,nbhd=24,knn_channels=None,xyz_dim=3,chin=64,**kwargs):
#         super().__init__()
#         self.ds_frac = ds_frac
#         self.knn_channels = knn_channels
#         self.mapping = PointConvBase(chin,xyz_dim,nbhd,xyz_dim,knn_channels,**kwargs)
#     def forward(self,x,coords_only=False):
#         # BCN representation assumed
#         coords,values = x
#         if self.ds_frac==1:
#             if coords_only: return coords
#             else: return x
#         coords = coords
#         values = values
#         num_downsampled_points = int(np.round(coords.shape[1]*self.ds_frac))
#         #key = pthash(coords[:,:,:self.knn_channels])
#         #if key not in self.subsample_lookup:# or True:
#             #print("Cache miss")
#         #    self.subsample_lookup[key] = farthest_point_sample(coords, num_downsampled_points)
#         fps_idx = farthest_point_sample(coords, num_downsampled_points)#self.subsample_lookup[key]

#         new_coords = index_points(coords,fps_idx)
#         new_values = index_points(values,fps_idx)
#         self.inp_coords = new_coords[0].cpu().data.numpy()
#         #print(new_values.shape,new_coords.shape)
#         offset = .03*self.mapping(x,new_coords)
#         self.offset = offset[0].cpu().data.numpy()
#         new_coords = new_coords + offset
#         self.out_coords = new_coords[0].cpu().data.numpy()
#         if coords_only: return new_coords
        
#         return new_coords,new_values
    # def log_data(self,logger,step,name):
    #     #print("log_data called")
    #     fig = plt.figure()
    #     ax = plt.axes(projection='3d')
    #     x,y,z = self.inp_coords.T
    #     dx,dy,dz = self.offset.T
    #     ax.cla()
    #     ax.scatter(x,y,z,c=z)
    #     ax.quiver(x,y,z,dx,dy,dz)#,length = mag)
    #     #ax.scatter(xp,yp,zp,c=zp)
    #     fig.canvas.draw()
    #     plt.show()


# class CoordinatePointConv(PointConvBase):
#     def __init__(self,*args,coordmap=Coordinates(),**kwargs):
#         super().__init__(*args,**kwargs)
#         self.coordmap = coordmap
#         self.net = WeightNet(self.xyz_dim+self.coordmap.embed_dim, self.cicm_co,hidden_unit = (32, 32))
#     def get_embedded_group_elems(self,nbhd_xyz,output_xyz):
#         return self.coordmap(nbhd_xyz) - self.coordmap(output_xyz)




