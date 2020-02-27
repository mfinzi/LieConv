import copy
import torch
import torch.nn as nn
from oil.utils.utils import Eval
from oil.model_trainers import Trainer
from lie_conv.hamiltonian import HamiltonianDynamics,EuclideanK
from lie_conv.lieConv import pConvBNrelu, PointConv, Pass, Swish, LieResNet
from lie_conv.moleculeTrainer import BottleBlock, GlobalPool
from lie_conv.utils import Expression, export, Named
import numpy as np
from torchdiffeq import odeint
from lie_conv.lieGroups import T

class Partial(nn.Module):
    def __init__(self,module,*args,**kwargs):
        super().__init__()
        self.module = module
        self.args = args
        self.kwargs = kwargs
    def forward(self,*x):
        self.module.nfe +=1
        return self.module(*x,*self.args,**self.kwargs)

@export
class IntegratedDynamicsTrainer(Trainer):
    """ Model should specify the dynamics, mapping from t,z,sysP -> dz/dt"""
    def __init__(self, *args, traj_data=None, tol=1e-4, **kwargs):
        super().__init__(*args,**kwargs)
        self.hypers['tol'] = tol
        self.num_mbs = 0
        self.traj_data = traj_data
        self.ckpt = (-1, copy.deepcopy(self.model.state_dict()))

    def _rollout_model(self, model, z0, ts, sys_params):
        """Parameters
        ----------
        model: torch.nn.Module
        z0: torch.Tensor, [batch_size, z_dim]
        ts: [batch_size, traj_len]
        sys_params: [batch_size, param_dim, num_params]

        Returns
        -------
        pred_zs: torch.Tensor, [batch_size, traj_len, z_dim]"""
        dynamics = Partial(model, sysP=sys_params)
        zs = odeint(dynamics, z0, ts[0], rtol=self.hypers['tol'], method='rk4')
        return zs.permute(1, 0, 2)

    def loss(self, minibatch, model=None):
        """ Standard cross-entropy loss """
        (z0, sys_params, ts), true_zs = minibatch
        model = self.model if model is None else model
        pred_zs = self._rollout_model(model, z0, ts, sys_params)
        self.num_mbs += 1
        return (pred_zs - true_zs).pow(2).mean()

    def _get_rollout_mse(self):
        assert self.traj_data is not None
        ts, true_zs, sys_params = self.traj_data
        z0 = true_zs[:, 0]
        with Eval(self.model), torch.no_grad():
            pred_zs = self._rollout_model(self.model, z0, ts, sys_params)
        return (pred_zs - true_zs).pow(2).mean().item()

    def metrics(self, loader):
        mse = lambda mb: self.loss(mb).cpu().data.numpy()
        return {'MSE':self.evalAverageMetrics(loader,mse)}

    def logStuff(self, step, minibatch=None):
        self.logger.add_scalars('info', {'nfe': self.model.nfe/(max(self.num_mbs, 1e-3))}, step)
        super().logStuff(step, minibatch)
        idx = self.logger.scalar_frame['val_MSE'].values.argmin()
        if not idx == self.ckpt[0]:
            self.ckpt = (idx, copy.deepcopy(self.model.state_dict()))

        
# class pConvLNswish(nn.Module):
#     def __init__(self,in_channels,out_channels,**kwargs):
#         super().__init__()
#         self.in_channels=in_channels
#         self.out_channels = out_channels
#         self.pconv = PointConv(in_channels,out_channels,**kwargs)
#         self.norm = MaskBatchNormNd(out_channels)
#     def forward(self,inp):
#         xyz,vals = self.pconv(inp)
#         if True: return xyz,vals # shortcut and ignore layernorm for now
#         bnc_vals = vals#.permute(0,2,1)
#         normed_vals = self.norm(bnc_vals)#.view(-1,self.out_channels))
#         return xyz,normed_vals#.view(*bnc_vals.shape)#.permute(0,2,1)
def logspace(a,b,k):
    return np.exp(np.linspace(np.log(a),np.log(b),k))

@export
class LieConvNetT2(nn.Module,metaclass=Named):
    """
    pointconvnet to model the potential function
    """
    def __init__(self, num_targets=1,k=64,num_layers=3,sys_dim=2,bn=False,**kwargs):
        super().__init__()
        assert num_targets <=1, "regression problem"
        chs = np.round(logspace(k,4*k,num_layers+1)).astype(int)
        chs[0] = sys_dim
        self.net = nn.Sequential(
            *[pConvBNrelu(chs[i],chs[i+1],ds_frac=1,nbhd=np.inf,act='swish',bn=bn,
                                xyz_dim=2,**kwargs) for i in range(num_layers)],
            Expression(lambda u:u[1].mean(1)),
            nn.Linear(chs[-1],num_targets)
        )
        self.nfe=0
    def compute_V(self,x):
        """ Input is a canonical position variable and the system parameters,
            shapes (bs, n,d) and (bs,n,c)"""
        q,sys_params = x
        mask = ~torch.isnan(q[...,0])
        v = self.net((q,sys_params,mask)).squeeze(-1)
        return v

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
class LieResNetT2(LieConvNetT2):
    """
    pointconvnet to model the potential function
    """
    def __init__(self, num_targets=1,k=1024,num_layers=3, d=2, sys_dim=2,bn=False):
        super().__init__()
        assert num_targets <=1, "regression problem"
        conv = lambda chin,chout: PointConv(chin, chout, nbhd=np.inf, ds_frac=1, bn=bn, 
                                   act='swish', mean=False, xyz_dim=d)
        self.net = nn.Sequential(
            Pass(nn.Linear(sys_dim,k),dim=1), #embedding layer
            *[BottleBlock(k,k,conv,bn=bn,act='swish')
                                for _ in range(num_layers)],
            Pass(nn.Linear(k,k//2),dim=1),
            Pass(Swish(),dim=1),  
            GlobalPool(mean=True),#mean), 
            nn.Linear(k//2,num_targets)
        )

@export
class HLieResNet(LieResNet):
    def __init__(self,d=2,sys_dim=2,bn=False,num_layers=4,group=T(2),k=384,knn=False,nbhd=100,mean=True,center=True,**kwargs):
        super().__init__(chin=sys_dim,ds_frac=1,num_layers=num_layers,nbhd=nbhd,mean=mean,bn=bn,xyz_dim=d,
                        group=group,fill=1.,k=k,num_outputs=1,cache=True,knn=knn,**kwargs)
        self.nfe=0
        self.center = center
    def compute_V(self,x):
        """ Input is a canonical position variable and the system parameters,
            shapes (bs, n,d) and (bs,n,c)"""
        q,sys_params = x
        mask = ~torch.isnan(q[...,0])
        v = super().forward((q,sys_params,mask)).squeeze(-1)
        return v

    def compute_H(self,z,sys_params):
        """ computes the hamiltonian, inputs (bs,2nd), (bs,n,c)"""
        m = sys_params[...,0] # assume the first component encodes masses
        #print("in H",z.shape,sys_params.shape)
        D = z.shape[-1] # of ODE dims, 2*num_particles*space_dim
        q = z[:,:D//2].reshape(*m.shape,-1)
        p = z[:,D//2:].reshape(*m.shape,-1)
        if self.center: q = q-q.mean(1,keepdims=True)#(m.unsqueeze(-1)*q).sum(dim=1,keepdims=True)/m.sum(1,keepdims=True).unsqueeze(-1)
        
        T=EuclideanK(p,m)
        V =self.compute_V((q,sys_params))
        return T+V
    
    def forward(self,t,z,sysP,wgrad=True):
        dynamics = HamiltonianDynamics(lambda t,z: self.compute_H(z,sysP),wgrad=wgrad)
        return dynamics(t,z)


@export
class FLieResnet(LieResNet):
    def __init__(self,d=2,sys_dim=2,bn=False,num_layers=4,group=T(2),k=384,knn=False,nbhd=100,mean=True,**kwargs):
        super().__init__(chin=sys_dim+d,ds_frac=1,num_layers=num_layers,nbhd=nbhd,mean=mean,bn=bn,xyz_dim=d,
                        group=group,fill=1.,k=k,num_outputs=2*d,cache=True,knn=knn,pool=False,**kwargs)
        self.nfe=0
    
    def forward(self,t,z,sysP,wgrad=True):
        m = sysP[...,0] # assume the first component encodes masses
        #print("in H",z.shape,sys_params.shape)
        D = z.shape[-1] # of ODE dims, 2*num_particles*space_dim
        q = z[:,:D//2].reshape(*m.shape,-1)
        p = z[:,D//2:].reshape(*m.shape,-1)
        q = q-q.mean(1,keepdims=True)#(m.unsqueeze(-1)*q).sum(dim=1,keepdims=True)/m.sum(1,keepdims=True).unsqueeze(-1)
        #q,sys_params = x
        bs,n,d = q.shape
        mask = ~torch.isnan(q[...,0])
        values = torch.cat([sysP,p],dim=-1)
        F = super().forward((q,values,mask)) #(bs,n,2d)
        flat_qdot = F[:,:,:d].reshape(bs,D//2)
        flat_pdot = F[:,:,d:].reshape(bs,D//2)
        dynamics = torch.cat([flat_qdot,flat_pdot],dim=-1)
        return dynamics



def FCswish(chin,chout):
    return nn.Sequential(nn.Linear(chin,chout),Swish())

@export
class FCHamNet(LieConvNetT2):
    def __init__(self, num_targets=1,k=150,num_layers=4,sys_dim=2, d=2):
        super().__init__()
        num_particles=6
        chs = [num_particles*(d+sys_dim)]+num_layers*[k]
        self.net = nn.Sequential(
            *[FCswish(chs[i],chs[i+1]) for i in range(num_layers)],
            nn.Linear(chs[-1],num_targets)
        )
    def compute_V(self,x):
        q,sys_params = x
        mean_subbed = (q-q.mean(1,keepdims=True),sys_params)
        return self.net(torch.cat(mean_subbed,dim=-1).reshape(q.shape[0],-1)).squeeze(-1)

@export
class HFC(FCHamNet): pass

@export
class RawDynamicsNet(nn.Module,metaclass=Named):
    def __init__(self, d=2,k=300,num_layers=4,sys_dim=2,**kwargs):
        super().__init__()
        num_particles=6
        chs = [num_particles*(2*d+sys_dim)]+num_layers*[k]
        self.net = nn.Sequential(
            *[FCswish(chs[i],chs[i+1]) for i in range(num_layers)],
            nn.Linear(chs[-1],2*d*num_particles)
        )
        self.nfe=0
    def forward(self,t,z,sysP,wgrad=True):
        m = sysP[...,0]
        D = z.shape[-1]
        q = z[:,:D//2].reshape(*m.shape,-1)
        p = z[:,D//2:]
        zm = torch.cat(((q - q.mean(1,keepdims=True)).reshape(z.shape[0],-1),p,sysP.reshape(z.shape[0],-1)),dim=1)
        return self.net(zm)

@export
class FC(RawDynamicsNet): pass
