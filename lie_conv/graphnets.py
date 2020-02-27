import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear as Lin
from torch_scatter import scatter_add
from torch_geometric.nn import MetaLayer
from lie_conv.utils import Named, export
from lie_conv.hamiltonian import HamiltonianDynamics,EuclideanK
from lie_conv.lieConv import Swish


class EdgeModel(torch.nn.Module):
    def __init__(self,in_dim,k=64):
        super().__init__()
        self.edge_mlp = Seq(Lin(in_dim, k), Swish(), Lin(k, k), Swish())

    def forward(self, src, dest, edge_attr, u, batch):
        # source, target: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        out = torch.cat([src, dest, edge_attr, u[batch]], 1)
        return self.edge_mlp(out)

class NodeModel(torch.nn.Module):
    def __init__(self,in_dim,k=64):
        super().__init__()
        self.node_mlp = Seq(Lin(in_dim, k), Swish(), Lin(k, k), Swish())

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        aggregated_edges = scatter_add(edge_attr, col, dim=0, dim_size=x.size(0))
        inputs = torch.cat([x, aggregated_edges, u[batch]], dim=1)
        return self.node_mlp(inputs)

class GlobalModel(torch.nn.Module):
    def __init__(self,in_dim,k=64):
        super().__init__()
        self.global_mlp = Seq(Lin(in_dim, k), Swish(), Lin(k, k), Swish())

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        agg_edges = scatter_add(edge_attr, batch[col], dim=0)
        agg_nodes = scatter_add(x, batch, dim=0)
        inputs = torch.cat([u, agg_edges, agg_nodes], dim=1)
        return self.global_mlp(inputs)

class GNlayer(torch.nn.Module):
    def __init__(self,in_dim,k):
        super().__init__()
        if isinstance(in_dim,tuple):
            nd,ed,gd = in_dim
        else:
            nd = ed = gd = in_dim
        self.layer = MetaLayer(EdgeModel(2*nd+ed+gd,k),
                               NodeModel(nd+k+gd,k),
                               GlobalModel(k+k+gd,k))
    def forward(self,z):
        v, e, u, edge_index, batch = z
        vp,ep,up = self.layer(v,edge_index,e,u,batch)
        return (vp, ep, up, edge_index, batch)

@export
class OGN(torch.nn.Module,metaclass=Named):
    def __init__(self,d=2,sys_dim=2,k=64,num_layers=1):
        super().__init__()
        self.gnlayers = nn.Sequential(
            GNlayer((2*d+sys_dim,1,1),k),
            *[GNlayer(k,k) for _ in range(num_layers-1)])
        #self.linear = nn.Linear(k,2*d)
        self.qlinear = nn.Linear(k,d)
        self.plinear = nn.Linear(k,d)
        self.nfe=0

    def featurize(self,z,sys_params):
        """z (bs,n,d) sys_params (bs,n,c) """
        # mask = torch.isnan(z)
        #z_zeros = torch.where(mask,z,torch.zeros_like(z))
        #sys_params_zeros = torch.where(mask[...,:1],sys_params,torch.zeros_like(sys_params))
        D = z.shape[-1]
        q = z[:,:D//2].reshape(*sys_params.shape[:-1],-1)
        p = z[:,D//2:].reshape(*sys_params.shape[:-1],-1)
        x = torch.cat([q - q.mean(1,keepdims=True),p,sys_params],dim=-1)
        bs,n,_ = x.shape
        cols = (torch.arange(n)[:,None]*torch.ones(n)[None,:])
        cols = (cols[None,:,:]+n*torch.arange(bs)[:,None,None]).to(q.device).long() #(bs,n,n) -> (bs*n*n)
        edge_index = cols.permute(0,2,1).reshape(-1), cols.reshape(-1)
        batch = (torch.arange(bs).to(q.device)[:,None]+torch.zeros(n).to(q.device)[None,:]).reshape(-1)
        e = torch.ones(bs*n*n,1).type(z.dtype).to(q.device) # edge level features
        v = x.reshape(bs*n,-1) # node level features
        u = torch.ones(bs,1).type(z.dtype).to(q.device) # global features
        return (v,e,u,edge_index,batch.long())

    def forward(self,t,z,sysP,wgrad=True):
        self.nfe+=1
        # (bs*n,d+c), (2, bs*n*n), (bs*n*n,1), (bs,1), (n)
        z = self.featurize(z,sysP) 
        vp,ep,up,_,_ = self.gnlayers(z) # (bs*n,k), (bs*n*n,k), (bs,k)
        #velocities = self.linear(vp) # (bs*n, 2d)
        #dynamics = velocities.reshape(up.shape[0],-1)
        bs = up.shape[0]
        flat_qdot = self.qlinear(vp).reshape(bs,-1)
        flat_pdot = self.plinear(vp).reshape(bs,-1)
        dynamics = torch.cat([flat_qdot,flat_pdot],dim=-1)
        return dynamics

@export
class HOGN(OGN):
    def __init__(self,d=2,sys_dim=2,k=64,num_layers=1):
        super().__init__(d,sys_dim,k,num_layers)
        self.linear = nn.Linear(k,1)

    def compute_H(self,z,sys_params):
        """ computes the hamiltonian, inputs (bs,2nd), (bs,n,c)"""
        m = sys_params[...,0] # assume the first component encodes masses
        z = self.featurize(z,sys_params) 
        vp,ep,up,_,_ = self.gnlayers(z) # (bs*n,k), (bs*n*n,k), (bs,k)
        energy = self.linear(up) # (bs,1)
        return energy.squeeze(-1)
    
    def forward(self,t,z,sysP,wgrad=True):
        dynamics = HamiltonianDynamics(lambda t,z: self.compute_H(z,sysP),wgrad=wgrad)
        return dynamics(t,z)

@export
class VOGN(OGN):
    def __init__(self,d=2,sys_dim=2,k=64,num_layers=1):
        super().__init__()
        self.gnlayers = nn.Sequential(
            GNlayer((d+sys_dim,1,1),k),
            *[GNlayer(k,k) for _ in range(num_layers-1)])
        self.linear = nn.Linear(k,1)
        self.nfe=0

    def featurize(self,q,sys_params):
        """z (bs,n,d) sys_params (bs,n,c) """
        # mask = torch.isnan(z)
        #z_zeros = torch.where(mask,z,torch.zeros_like(z))
        #sys_params_zeros = torch.where(mask[...,:1],sys_params,torch.zeros_like(sys_params))
        x = torch.cat([q - q.mean(1,keepdims=True),sys_params],dim=-1)
        bs,n,_ = x.shape
        cols = (torch.arange(n)[:,None]*torch.ones(n)[None,:])
        cols = (cols[None,:,:]+n*torch.arange(bs)[:,None,None]).to(q.device).long() #(bs,n,n) -> (bs*n*n)
        edge_index = cols.permute(0,2,1).reshape(-1), cols.reshape(-1)
        batch = (torch.arange(bs).to(q.device)[:,None]+torch.zeros(n).to(q.device)[None,:]).reshape(-1)
        e = torch.ones(bs*n*n,1).type(q.dtype).to(q.device) # edge level features
        v = x.reshape(bs*n,-1) # node level features
        u = torch.ones(bs,1).type(q.dtype).to(q.device) # global features
        return (v,e,u,edge_index,batch.long())
    def compute_V(self,x):
        """ Input is a canonical position variable and the system parameters,
            shapes (bs, n,d) and (bs,n,c)"""
        q,sys_params = x
        z = self.featurize(q,sys_params)
        vp,ep,up,_,_ = self.gnlayers(z) # (bs*n,k), (bs*n*n,k), (bs,k)
        energy = self.linear(up) # (bs,1)
        return energy.squeeze(-1)
    def compute_H(self,z,sys_params):
        """ computes the hamiltonian, inputs (bs,2nd), (bs,n,c)"""
        m = sys_params[...,0] # assume the first component encodes masses
        #print("in H",z.shape,sys_params.shape)
        D = z.shape[-1] # of ODE dims, 2*num_particles*space_dim
        q = z[:,:D//2].reshape(*m.shape,-1)
        p = z[:,D//2:].reshape(*m.shape,-1)
        T=EuclideanK(p,m)
        V =self.compute_V((q,sys_params))
        return T+V
    
    def forward(self,t,z,sysP,wgrad=True):
        dynamics = HamiltonianDynamics(lambda t,z: self.compute_H(z,sysP),wgrad=wgrad)
        return dynamics(t,z)

@export
class MolecGN(nn.Module,metaclass=Named):
    def __init__(self,num_species,charge_scale,num_outputs=1,d=3,k=64,num_layers=1):
        super().__init__()
        self.gnlayers = nn.Sequential(
            GNlayer((d+num_species*3,1,1),k),
            *[GNlayer(k,k) for _ in range(num_layers-1)])
        self.linear = nn.Linear(k,num_outputs)
        self.charge_scale = charge_scale

    def featurize(self,mb):
        charges = (mb['charges']/self.charge_scale)
        c_vec = torch.stack([torch.ones_like(charges),charges,charges**2],dim=-1) # 
        one_hot_charges = (mb['one_hot'][:,:,:,None]*c_vec[:,:,None,:]).float().reshape(*charges.shape,-1) #(bs,n,5) (bs,n)
        atomic_coords = mb['positions'].float()
        x = torch.cat([one_hot_charges,atomic_coords],dim=-1)
        bs,n,_ = x.shape
        cols = (torch.arange(n)[:,None]*torch.ones(n)[None,:])
        cols = (cols[None,:,:]+n*torch.arange(bs)[:,None,None]).to(x.device).long() #(bs,n,n) -> (bs*n*n)
        edge_index = cols.permute(0,2,1).reshape(-1), cols.reshape(-1)
        batch = (torch.arange(bs).to(x.device)[:,None]+torch.zeros(n).to(x.device)[None,:]).reshape(-1)
        e = torch.ones(bs*n*n,1).type(x.dtype).to(x.device) # edge level features
        v = x.reshape(bs*n,-1) # node level features
        u = torch.ones(bs,1).type(x.dtype).to(x.device) # global features
        return (v,e,u,edge_index,batch.long())

    def forward(self,mb):
        x = self.featurize(mb)
        vp,ep,up,_,_ = self.gnlayers(x) # (bs*n,k), (bs*n*n,k), (bs,k)
        return self.linear(up).squeeze(-1)