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
    def __init__(self, *args, tol=1e-4, **kwargs):
        super().__init__(*args,**kwargs)
        self.hypers['tol'] = tol
        self.num_mbs = 0

    def _rollout_model(self, z0, ts, sys_params):
        """ inputs [z0: (bs, z_dim), ts: (bs, T), sys_params: (bs, n, c)]
            outputs pred_zs: (bs, T, z_dim) """
        dynamics = Partial(self.model, sysP=sys_params)
        zs = odeint(dynamics, z0, ts[0], rtol=self.hypers['tol'], method='rk4')
        return zs.permute(1, 0, 2)

    def loss(self, minibatch):
        """ Standard cross-entropy loss """
        (z0, sys_params, ts), true_zs = minibatch
        pred_zs = self._rollout_model(z0, ts, sys_params)
        self.num_mbs += 1
        return (pred_zs - true_zs).pow(2).mean()

    def get_rollout_mse(self,traj_data):
        ts, true_zs, sys_params = traj_data
        z0 = true_zs[:, 0]
        with Eval(self.model), torch.no_grad():
            pred_zs = self._rollout_model(z0, ts, sys_params)
        return (pred_zs - true_zs).pow(2).mean().item()

    def metrics(self, loader):
        mse = lambda mb: self.loss(mb).cpu().data.numpy()
        return {'MSE':self.evalAverageMetrics(loader,mse)}

    def logStuff(self, step, minibatch=None):
        self.logger.add_scalars('info', {'nfe': self.model.nfe/(max(self.num_mbs, 1e-3))}, step)
        super().logStuff(step, minibatch)

def logspace(a,b,k):
    return np.exp(np.linspace(np.log(a),np.log(b),k))

def FCswish(chin,chout):
    return nn.Sequential(nn.Linear(chin,chout),Swish())

@export
class FC(nn.Module,metaclass=Named):
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

class HNet(nn.Module,metaclass=Named): # abstract Hamiltonian network class
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
class HFC(HNet):
    def __init__(self, num_targets=1,k=150,num_layers=4,sys_dim=2, d=2):
        super().__init__()
        num_particles=6
        chs = [num_particles*(d+sys_dim)]+num_layers*[k]
        self.net = nn.Sequential(
            *[FCswish(chs[i],chs[i+1]) for i in range(num_layers)],
            nn.Linear(chs[-1],num_targets)
        )
        self.nfe=0
    def compute_V(self,x):
        """ Input is a canonical position variable and the system parameters,
            shapes (bs, n,d) and (bs,n,c)"""
        q,sys_params = x
        mean_subbed = (q-q.mean(1,keepdims=True),sys_params)
        return self.net(torch.cat(mean_subbed,dim=-1).reshape(q.shape[0],-1)).squeeze(-1)

@export
class HLieResNet(LieResNet,HNet):
    def __init__(self,d=2,sys_dim=2,bn=False,num_layers=4,group=T(2),k=384,knn=False,nbhd=100,mean=True,center=True,**kwargs):
        super().__init__(chin=sys_dim,ds_frac=1,num_layers=num_layers,nbhd=nbhd,mean=mean,bn=bn,xyz_dim=d,
                        group=group,fill=1.,k=k,num_outputs=1,cache=True,knn=knn,**kwargs)
        self.nfe=0
        self.center = center
    def forward(self,t,z,sysP,wgrad=True):
        dynamics = HamiltonianDynamics(lambda t,z: self.compute_H(z,sysP),wgrad=wgrad)
        return dynamics(t,z)
    def compute_V(self,x):
        """ Input is a canonical position variable and the system parameters,
            shapes (bs, n,d) and (bs,n,c)"""
        q,sys_params = x
        mask = ~torch.isnan(q[...,0])
        if self.center: q = q-q.mean(1,keepdims=True)
        return super().forward((q,sys_params,mask)).squeeze(-1)

@export
class FLieResnet(LieResNet): # An (equivariant) lieConv network that models the dynamics directly
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





