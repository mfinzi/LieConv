import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class HamiltonianDynamics(nn.Module):
    """ Defines the dynamics given a hamiltonian. If wgrad=True, the dynamics can be backproped."""
    def __init__(self,H,wgrad=False):
        super().__init__()
        self.H = H
        self.wgrad=wgrad
        self.nfe=0
    def forward(self,t,z):
        self.nfe+=1
        with torch.enable_grad():
            z = torch.zeros_like(z, requires_grad=True) + z
            D = z.shape[-1]
            h = self.H(t,z).sum() # elements in mb are independent, gives mb gradients
            rg = torch.autograd.grad(h,z,create_graph=self.wgrad)[0] # riemannian gradient
        sg = torch.cat([rg[:,D//2:],-rg[:,:D//2]],dim=-1) # symplectic gradient = SdH
        return sg


def EuclideanK(momentums, masses):
    """ Shape (bs,n,d), and (bs,n),
        standard \sum_n p_n^2/{2m_n} kinetic energy"""
    p_sq_norms = momentums.pow(2).sum(-1)
    kinetic_energy = (p_sq_norms / masses).sum(-1) / 2
    return kinetic_energy
    # return (p*(p/m[:,:,None])).sum(-1).sum(-1)/2

def KeplerV(positions, masses):
    """ Shape (bs,n,d), and (bs,n),
        Gravitational PE: -\sum_{jk} m_jm_k/{\|q_j-q_k\|}"""
    grav_const = 1
    n = masses.shape[-1]
    row_ind, col_ind = torch.tril_indices(n, n, offset=-1)
    moments = (masses.unsqueeze(1) * masses.unsqueeze(2))[:, row_ind, col_ind]
    pair_diff = (positions.unsqueeze(1) - positions.unsqueeze(2))[:, row_ind, col_ind]
    pair_dist = pair_diff.norm(dim=-1) + 1e-8
    potential_energy = -grav_const * (moments / pair_dist).sum(-1)
    return potential_energy

def KeplerH(z,m):
    """ with shapes (bs,2nd)"""
    bs, D = z.shape # of ODE dims, 2*num_particles*space_dim
    q = z[:,:D//2].reshape(*m.shape,-1)
    p = z[:,D//2:].reshape(*m.shape,-1)
    potential_energy = KeplerV(q, m)
    kinetic_energy = EuclideanK(p, m)
    assert potential_energy.shape[0] == bs
    assert kinetic_energy.shape[0] == bs
    return potential_energy + kinetic_energy

def SpringV(q,k):
    """ Potential for a bunch particles connected by springs with kij
        Shape (bs,n,d), and (bs,n,n)"""
    K = k[:,:,None]*k[:,None,:] #(bs,n,n)
    n = K.shape[-1]
    radial = (q[:,:,None,:] - q[:,None,:,:]).norm(dim=-1)**2 # (bs, n, n)
    potential = .5*(K*radial).sum(-1).sum(-1)
    return potential #(bs,n,n) -> (bs)

def SpringH(z,m,k):
    """ with shapes (bs,2nd)"""
    D = z.shape[-1] # of ODE dims, 2*num_particles*space_dim
    q = z[:,:D//2].reshape(*m.shape,-1)
    p = z[:,D//2:].reshape(*m.shape,-1)
    return EuclideanK(p,m) + SpringV(q,k)

def BallV(q,r):
    """ Potential for a bunch of (almost) rigid balls and walls, each ball has radius r"""
    n = r.shape[-1]
    thresh = 0.1
    barrier = lambda dist: .5*(torch.exp(1/(dist-thresh*1.05) - 50*(dist-thresh))).sum(-1)#50*((dist-thresh)**2).sum(-1)#.5*(torch.exp(1/(dist-thresh*1.05))/dist).sum(-1)#1/(dist-thresh*1.05)
    separation = (q[:,:,None,:] - q[:,None,:,:]).norm(dim=-1)
    sum_r = r[:,:,None]+r[:,None,:]
    touching = (separation-sum_r < thresh)
    energy = barrier(separation[touching]-sum_r[touching])
    for i in range(q.shape[-1]):
        ld = q[:,:,i]+1-r
        rd = 1-q[:,:,i]-r
        energy += barrier(ld[ld<thresh])+barrier(rd[rd<thresh])
    return energy

def BallH(z,m,r):
    D = z.shape[-1] # of ODE dims, 2*num_particles*space_dim
    q = z[:,:D//2].reshape(*m.shape,-1)
    p = z[:,D//2:].reshape(*m.shape,-1)
    return EuclideanK(p,m) + BallV(q,r)


# TODO:
# Make animation plots look nicer. Why are there leftover points on the trails?
class Animation2d(object):
    def __init__(self, qt, ms=None, box_lim=(-1, 1)):
        if ms is None: ms = len(qt)*[6]
        self.qt = qt
        self.fig = plt.figure()
        self.ax = self.fig.add_axes([0, 0, 1, 1])#axes(projection='3d')
        self.ax.set_xlim(box_lim)
        self.ax.set_ylim(box_lim)
        self.lines = sum([self.ax.plot([],[],'-') for particle in self.qt],[])
        self.pts = sum([self.ax.plot([],[],'o',ms=ms[i]) for i in range(len(self.qt))],[])
    def init(self):
        for line,pt in zip(self.lines,self.pts):
            line.set_data([], [])
            pt.set_data([], [])
        return self.lines + self.pts
    def update(self,i=0):
        for line, pt, trajectory in zip(self.lines,self.pts,self.qt):
            x,y = trajectory[:,:i]
            line.set_data(x,y)
            pt.set_data(x[-1:], y[-1:])
        #self.fig.clear()
        self.fig.canvas.draw()
        return self.lines+self.pts
    def animate(self):
        return animation.FuncAnimation(self.fig,self.update,frames=self.qt.shape[-1],
                                       interval=33,init_func=self.init,blit=True)
    
class Animation3d(object):
    def __init__(self,qt,ms=None, box_lim=(-1, 1)):
        if ms is None: ms = len(qt)*[6]
        self.qt = qt
        self.fig = plt.figure()
        self.ax = self.fig.add_axes([0, 0, 1, 1],projection='3d')#axes(projection='3d')
        self.ax.set_xlim3d(box_lim)
        self.ax.set_ylim3d(box_lim)
        self.ax.set_zlim3d(box_lim)
        self.lines = sum([self.ax.plot([],[],[],'-') for _ in self.qt],[])
        self.pts = sum([self.ax.plot([],[],[],'o',ms=ms[i]) for i in range(len(self.qt))],[])
    def init(self):
        for line,pt in zip(self.lines,self.pts):
            line.set_data([], [])
            line.set_3d_properties([])
            pt.set_data([], [])
            pt.set_3d_properties([])
        return self.lines + self.pts
    def update(self,i=0):
        for line, pt, trajectory in zip(self.lines,self.pts,self.qt):
            x,y,z = trajectory[:,:i]
            line.set_data(x,y)
            line.set_3d_properties(z)
            pt.set_data(x[-1:], y[-1:])
            pt.set_3d_properties(z[-1:])
        #self.fig.clear()
        self.fig.canvas.draw()
        return self.lines+self.pts
    def animate(self):
        return animation.FuncAnimation(self.fig,self.update,frames=self.qt.shape[-1],
                                       interval=33,init_func=self.init,blit=True)
def AnimationNd(n):
    if n==2: return Animation2d
    elif n==3: return Animation3d
    else: assert False, "No animation for d={}".format(n)