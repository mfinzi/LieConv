import torch
import numpy as np
from lie_conv.utils import export, Named

@export
def norm(x,dim):
    return (x**2).sum(dim=dim).sqrt()

class LieGroup(object,metaclass=Named):
    """ The abstract Lie Group requiring additional implementation of exp,log, and lifted_elems
        to use as a new group for LieConv. rep_dim,lie_dim,q_dim should additionally be specified."""
    rep_dim = NotImplemented # dimension on which G acts. (e.g. 2 for SO(2))
    lie_dim = NotImplemented # dimension of the lie algebra of G. (e.g. 1 for SO(2))
    q_dim = NotImplemented # dimension which the quotient space X/G is embedded. (e.g. 1 for SO(2) acting on R2)
    
    def __init__(self,alpha=.2):
        super().__init__()
        self.alpha=alpha

    def exp(self,a):
        """ Computes (matrix) exponential Lie algebra elements (in a given basis).
            ie out = exp(\sum_i a_i A_i) where A_i are the exponential generators of G.
            Input: [a (*,lie_dim)] where * is arbitrarily shaped
            Output: [exp(a) (*,rep_dim,rep_dim)] returns the matrix for each."""
        raise NotImplementedError
    
    def log(self,u):
        """ Computes (matrix) logarithm for collection of matrices and converts to Lie algebra basis.
            Input [u (*,rep_dim,rep_dim)]
            Output [coeffs of log(u) in basis (*,d)] """
        raise NotImplementedError
    
    def lifted_elems(self,xyz,nsamples):
        """ Takes in coordinates xyz and lifts them to Lie algebra elements a (in basis)
            and embedded orbit identifiers q. For groups where lifting is multivalued
            specify nsamples>1 as number of lifts to do for each point.
            Inputs: [xyz (*,n,rep_dim)],[mask (*,n)], [mask (int)]
            Outputs: [a (*,n*nsamples,lie_dim)],[q (*,n*nsamples,q_dim)]"""
        raise NotImplementedError
    
    def inv(self,g):
        """ We can compute the inverse of elements g (*,rep_dim,rep_dim) as exp(-log(g))"""
        return self.exp(-self.log(g))

    def distance(self,abq_pairs):
        """ Compute distance of size (*) from [abq_pairs (*,lie_dim+2*q_dim)].
            Simply computes alpha*norm(log(v^{-1}u)) +(1-alpha)*norm(q_a-q_b),
            combined distance from group element distance and orbit distance."""
        ab_dist = norm(abq_pairs[...,:self.lie_dim],dim=-1)
        qa = abq_pairs[...,self.lie_dim:self.lie_dim+self.q_dim]
        qb = abq_pairs[...,self.lie_dim+self.q_dim:self.lie_dim+2*self.q_dim]
        qa_qb_dist = norm(qa-qb,dim=-1)
        return ab_dist*self.alpha + (1-self.alpha)*qa_qb_dist
    
    def lift(self,x,nsamples,**kwargs):
        """assumes p has shape (*,n,2), vals has shape (*,n,c), mask has shape (*,n)
            returns (a,v) with shapes [(*,n*nsamples,lie_dim),(*,n*nsamples,c)"""
        p,v,m = x
        expanded_a,expanded_q = self.lifted_elems(p,nsamples,**kwargs) # (bs,n*ns,d), (bs,n*ns,qd)
        nsamples = expanded_a.shape[-2]//m.shape[-1]
        # expand v and mask like q
        expanded_v = v[...,None,:].repeat((1,)*len(v.shape[:-1])+(nsamples,1)) # (bs,n,c) -> (bs,n,1,c) -> (bs,n,ns,c)
        expanded_v = expanded_v.reshape(*expanded_a.shape[:-1],v.shape[-1]) # (bs,n,ns,c) -> (bs,n*ns,c)
        expanded_mask = m[...,None].repeat((1,)*len(v.shape[:-1])+(nsamples,)) # (bs,n) -> (bs,n,ns)
        expanded_mask = expanded_mask.reshape(*expanded_a.shape[:-1]) # (bs,n,ns) -> (bs,n*ns)
        # convert from elems to pairs
        paired_a = self.elems2pairs(expanded_a) #(bs,n*ns,d) -> (bs,n*ns,n*ns,d)
        if expanded_q is not None:
            q_in = expanded_q.unsqueeze(-2).expand(*paired_a.shape[:-1],1)
            q_out = expanded_q.unsqueeze(-3).expand(*paired_a.shape[:-1],1)
            embedded_locations = torch.cat([paired_a,q_in,q_out],dim=-1)
        else:
            embedded_locations = paired_a
        return (embedded_locations,expanded_v,expanded_mask)
    
    def expand_like(self,v,m,a):
        nsamples = a.shape[-2]//m.shape[-1]
        expanded_v = v[...,None,:].repeat((1,)*len(v.shape[:-1])+(nsamples,1)) # (bs,n,c) -> (bs,n,1,c) -> (bs,n,ns,c)
        expanded_v = expanded_v.reshape(*a.shape[:2],v.shape[-1]) # (bs,n,ns,c) -> (bs,n*ns,c)
        expanded_mask = m[...,None].repeat((1,)*len(v.shape[:-1])+(nsamples,)) # (bs,n) -> (bs,n,ns)
        expanded_mask = expanded_mask.reshape(*a.shape[:2]) # (bs,n,ns) -> (bs,n*ns)
        return expanded_v, expanded_mask
    
    def elems2pairs(self,a):
        """ computes log(e^-b e^a) for all a b pairs along n dimension of input.
            inputs: [a (bs,n,d)] outputs: [pairs_ab (bs,n,n,d)] """
        vinv = self.exp(-a.unsqueeze(-3))
        u = self.exp(a.unsqueeze(-2))
        return self.log(vinv@u)    # ((bs,1,n,d) -> (bs,1,n,r,r))@((bs,n,1,d) -> (bs,n,1,r,r))

    def BCH(self,a,b,order=2):
        """ Baker Campbell Hausdorff formula"""
        assert order <= 4, "BCH only supported up to order 4"
        B = self.bracket
        z = a+b
        if order==1: return z
        ab = B(a,b)
        z += (1/2)*ab
        if order==2: return z
        aab = B(a,ab)
        bba = B(b,-ab)
        z += (1/12)*(aab+bba)
        if order==3: return z
        baab = B(b,aab)
        z += -(1/24)*baab
        return z
    
    def bracket(self,a,b):
        """Computes the lie bracket between a and b, assumes a,b expressed as vectors"""
        A = self.components2matrix(a)
        B = self.components2matrix(b)
        return self.matrix2components(A@B-B@A)

    def __str__(self):
        return f"{self.__class__}({self.alpha})" if self.alpha!=.2 else f"{self.__class__}"
    def __repr__(self):
        return str(self)

@export
def LieSubGroup(liegroup,generators):
    
    class subgroup(liegroup):
        
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.orig_dim = self.lie_dim
            self.lie_dim = len(generators)
            self.q_dim = self.orig_dim-len(generators)

        def exp(self,a_small):
            a_full = torch.zeros(*a_small.shape[:-1],self.orig_dim,
                        device=a_small.device,dtype=a_small.dtype)
            a_full[...,generators] = a_small
            return super().exp(a_full)
        
        def log(self,U):
            return super().log(U)[...,generators]
        
        def components2matrix(self,a_small):
            a_full = torch.zeros(*a_small.shape[:-1],self.orig_dim,
                         device=a_small.device,dtype=a_small.dtype)
            a_full[...,generators] = a_small
            return super().components2matrix(a_full)
        
        def matrix2components(self,A):
            return super().matrix2components(A)[...,generators]
        def lifted_elems(self,pt,nsamples=1):
            """ pt (bs,n,D) mask (bs,n), per_point specifies whether to
                use a different group element per atom in the molecule"""
            a_full,q = super().lifted_elems(pt,nsamples)
            a_sub = a_full[...,generators]
            complement_generators = list(set(range(self.orig_dim))-set(generators))
            new_qs = a_full[...,complement_generators]
            q_sub = torch.cat([q,new_qs],dim=-1) if q is not None else new_qs
            return a_sub,q_sub
        # def __str__(self):
        #     return f"Subgroup({str(liegroup)},{generators})"
    return subgroup

@export
class T(LieGroup):
    def __init__(self,k):
        """ Returns the k dimensional translation group. Assumes lifting from R^k"""
        super().__init__()
        self.q_dim = 0
        self.rep_dim = k # dimension on which G acts
        self.lie_dim = k # dimension that g is embedded into

    def lifted_elems(self,xyz,nsamples,**kwargs):
        assert nsamples==1, "Abelian group, no need for nsamples"
        return xyz,None
    
    def elems2pairs(self,a):
        deltas = a.unsqueeze(-2)-a.unsqueeze(-3)
        return deltas
    # def distance(self,embedded_pairs):
    #     return norm(embedded_pairs,dim=-1)

# Helper functions for analytic exponential maps. Uses taylor expansions near x=0
# See http://ethaneade.com/lie_groups.pdf for derivations.
thresh =7e-2
def sinc(x):
    """ sin(x)/x """
    x2=x*x
    usetaylor = (x.abs()<thresh)
    return torch.where(usetaylor,1-x2/6*(1-x2/20*(1-x2/42)),x.sin()/x)
def sincc(x):
    """ (1-sinc(x))/x^2"""
    x2=x*x
    usetaylor = (x.abs()<thresh)
    return torch.where(usetaylor,1/6*(1-x2/20*(1-x2/42*(1-x2/72))),(x-x.sin())/x**3)
def cosc(x):
    """ (1-cos(x))/x^2"""
    x2 = x*x
    usetaylor = (x.abs()<thresh)
    return torch.where(usetaylor,1/2*(1-x2/12*(1-x2/30*(1-x2/56))),(1-x.cos())/x**2)
def coscc(x):
    """  """
    x2 = x*x
    #assert not torch.any(torch.isinf(x2)), f"infs in x2 log"
    usetaylor = (x.abs()<thresh)
    texpand = 1/12*(1+x2/60*(1+x2/42*(1+x2/40)))
    costerm = (2*(1-x.cos())).clamp(min=1e-6)
    full = (1-x*x.sin()/costerm)/x**2 #Nans can come up here when cos = 1
    output = torch.where(usetaylor,texpand,full)
    return output

def sinc_inv(x):
    usetaylor = (x.abs()<thresh)
    texpand = 1+(1/6)*x**2 +(7/360)*x**4
    assert not torch.any(torch.isinf(texpand)|torch.isnan(texpand)),'sincinv texpand inf'+torch.any(torch.isinf(texpand))
    return torch.where(usetaylor,texpand,x/x.sin())

## Lie Groups acting on R2

@export
class SO2(LieGroup):
    lie_dim = 1
    rep_dim = 2
    q_dim = 1
    def exp(self,a):
        R = torch.zeros(*a.shape[:-1],2,2,device=a.device,dtype=a.dtype)
        sin = a[...,0].sin()
        cos = a[...,0].cos()
        R[...,0,0] = cos
        R[...,1,1] = cos
        R[...,0,1] = -sin
        R[...,1,0] = sin
        return R
    def log(self,R):
        return torch.atan2(R[...,1,0]-R[...,0,1],R[...,0,0]+R[...,1,1])[...,None]
    def components2matrix(self,a): # a: (*,lie_dim)
        A = torch.zeros(*a.shape[:-1],2,2,device=a.device,dtype=a.dtype)
        A[...,0,1] = -a[...,0]
        A[...,1,0] = a[...,0]
        return A
    def matrix2components(self,A): # A: (*,rep_dim,rep_dim)
        a = torch.zeros(*A.shape[:-1],1,device=A.device,dtype=A.dtype)
        a[...,:1] = (A[...,1,:1]-A[...,:1,1])/2
        return a
    def lifted_elems(self,pt,nsamples=1):
        """ pt (bs,n,D) mask (bs,n), per_point specifies whether to
            use a different group element per atom in the molecule"""
        assert nsamples==1, "Abelian group, no need for nsamples"
        bs,n,D = pt.shape[:3] # origin = [1,0]
        assert D==2, "Lifting from R^2 to SO(2) supported only"
        r = norm(pt,dim=-1).unsqueeze(-1)
        theta = torch.atan2(pt[...,1],pt[...,0]).unsqueeze(-1)
        return theta,r # checked that lifted_elem(v)@[0,1] = v
    def distance(self,abq_pairs):
        angle_pairs = abq_pairs[...,0]
        ra = abq_pairs[...,1]
        rb = abq_pairs[...,2]
        return angle_pairs.abs()*self.alpha + (1-self.alpha)*(ra-rb).abs()/(ra+rb+1e-3)

@export
class RxSO2(LieGroup):
    """ Rotation scaling group. Equivalent to log polar convolution."""
    lie_dim=2
    rep_dim=2
    q_dim=0
    def exp(self,a):
        logr = a[...,0]
        R = torch.zeros(*a.shape[:-1],2,2,device=a.device,dtype=a.dtype)
        rsin = logr.exp()*a[...,1].sin()
        rcos = logr.exp()*a[...,1].cos()
        R[...,0,0] = rcos
        R[...,1,1] = rcos
        R[...,0,1] = -rsin
        R[...,1,0] = rsin
        return R
    def log(self,R):
        rsin = (R[...,1,0]-R[...,0,1])/2
        rcos = (R[...,0,0]+R[...,1,1])/2
        theta = torch.atan2(rsin,rcos)
        r = (rsin**2+rcos**2).sqrt()
        return torch.stack([r.log(),theta],dim=-1)
    def lifted_elems(self,pt,nsamples=1):
        bs,n,D = pt.shape[:3] # origin = [1,0]
        assert D==2, "Lifting from R^2 to RxSO(2) supported only"
        r = norm(pt,dim=-1).unsqueeze(-1)
        theta = torch.atan2(pt[...,1],pt[...,0]).unsqueeze(-1)
        return torch.cat([r.log(),theta],dim=-1),None
    def distance(self,abq_pairs):
        angle_dist = abq_pairs[...,1].abs()
        r_dist = abq_pairs[...,0].abs()
        return angle_dist*self.alpha + (1-self.alpha)*r_dist

@export
class RxSQ(LieGroup):
    """ Rotation Squeeze group. Equivalent to log hyperbolic coordinate convolution.
        Acts on the positive orthant R2+."""
    lie_dim=2
    rep_dim=2
    q_dim=0
    def exp(self,a):
        raise NotImplementedError
    def log(self,R):
        raise NotImplementedError
    def lifted_elems(self,pt,nsamples=1):
        bs,n,D = pt.shape[:3] # origin = [1,0]
        assert nsamples==1, "Abelian group, no need for nsamples"
        assert D==2, "Lifting from R^2 to RxSQ supported only"
        lxy = pt.log()
        logs = (lxy[...,0]-lxy[...,1])/2
        logr = (lxy[...,0]+lxy[...,1])/2
        return torch.cat([logr,logs],dim=-1),None
    def distance(self,abq_pairs):
        s_dist = abq_pairs[...,1].abs()
        r_dist = abq_pairs[...,0].abs()
        return s_dist*self.alpha + (1-self.alpha)*r_dist
        
@export
class Rx(LieSubGroup(RxSO2,(0,))): pass
@export
class SQ(LieSubGroup(RxSQ,(1,))): pass
@export
class Tx(LieSubGroup(T,(0,))): pass
@export
class Ty(LieSubGroup(T,(1,))): pass

@export
class SE2(SO2):
    lie_dim = 3
    rep_dim = 3
    q_dim = 0
    def log(self,g):
        theta = super().log(g[...,:2,:2])
        I = torch.eye(2,device=g.device,dtype=g.dtype)
        K = super().components2matrix(torch.ones_like(theta))
        theta = theta.unsqueeze(-1)
        Vinv = (sinc(theta)/(2*cosc(theta)))*I - theta*K/2
        a = torch.zeros(g.shape[:-1],device=g.device,dtype=g.dtype)
        a[...,0] = theta[...,0,0]
        a[...,1:] = (Vinv@g[...,:2,2].unsqueeze(-1)).squeeze(-1)
        return a
    
    def exp(self,a):
        """ assumes that a is expanded in the basis [tx,ty,theta] of the lie algebra
            a should have shape (*,3)"""
        theta = a[...,0].unsqueeze(-1)
        I = torch.eye(2,device=a.device,dtype=a.dtype)
        K = super().components2matrix(torch.ones_like(a))
        theta = theta.unsqueeze(-1)
        V = sinc(theta)*I + theta*cosc(theta)*K
        g = torch.zeros(*a.shape[:-1],3,3,device=a.device,dtype=a.dtype)
        g[...,:2,:2] = theta.cos()*I+theta.sin()*K
        g[...,:2,2] = (V@a[...,1:].unsqueeze(-1)).squeeze(-1)
        g[...,2,2] = 1
        return g
    
    def components2matrix(self,a):
        """takes an element in the lie algebra expressed in the standard basis and
            expands to the corresponding matrix. a: (*,3)"""
        A = torch.zeros(*a.shape,3,device=a.device,dtype=a.dtype)
        A[...,2,:2] = a[...,1:]
        A[...,0,1] = a[...,0]
        A[...,1,0] = -a[...,0]
        return A
    
    def matrix2components(self,A):
        """takes an element in the lie algebra expressed as a matrix (*,3,3) and
            expresses it in the standard basis"""
        a = torch.zeros(*A.shape[:-1],device=A.device,dtype=A.dtype)
        a[...,1:] = A[...,:2,2]
        a[...,0] = (A[...,1,0]-A[...,0,1])/2
        return a
    
    def lifted_elems(self,pt,nsamples=1):
        #TODO: correctly handle masking, unnecessary for image data
        d=self.rep_dim
        # Sample stabilizer of the origin
        #thetas = (torch.rand(*p.shape[:-1],1).to(p.device)*2-1)*np.pi
        #thetas = torch.randn(nsamples)*2*np.pi - np.pi
        thetas = torch.linspace(-np.pi,np.pi,nsamples+1)[1:].to(pt.device)
        for _ in pt.shape[:-1]: # uniform on circle, but -pi and pi ar the same
            thetas=thetas.unsqueeze(0)
        thetas = thetas + torch.rand(*pt.shape[:-1],1).to(pt.device)*2*np.pi
        R = torch.zeros(*pt.shape[:-1],nsamples,d,d).to(pt.device)
        sin,cos = thetas.sin(),thetas.cos()
        R[...,0,0] = cos
        R[...,1,1] = cos
        R[...,0,1] = -sin
        R[...,1,0] = sin
        R[...,2,2] = 1
        # Get T(p)
        T = torch.zeros_like(R)
        T[...,0,0]=1
        T[...,1,1]=1
        T[...,2,2]=1
        T[...,:2,2] = pt.unsqueeze(-2)
        flat_a = self.log(T@R).reshape(*pt.shape[:-2],pt.shape[-2]*nsamples,d)
        return flat_a, None
    def distance(self,abq_pairs):
        d_theta = abq_pairs[...,0].abs()
        d_r = norm(abq_pairs[...,1:],dim=-1)
        return d_theta*self.alpha + (1-self.alpha)*d_r

## Lie Groups acting on R3

# Hodge star on R3
def cross_matrix(k):
    """Application of hodge star on R3, mapping Λ^1 R3 -> Λ^2 R3"""
    K = torch.zeros(*k.shape[:-1],3,3,device=k.device,dtype=k.dtype)
    K[...,0,1] = -k[...,2]
    K[...,0,2] = k[...,1]
    K[...,1,0] = k[...,2]
    K[...,1,2] = -k[...,0]
    K[...,2,0] = -k[...,1]
    K[...,2,1] = k[...,0]
    return K

def uncross_matrix(K):
    """Application of hodge star on R3, mapping Λ^2 R3 -> Λ^1 R3"""
    k = torch.zeros(*K.shape[:-1],device=K.device,dtype=K.dtype)
    k[...,0] = (K[...,2,1] - K[...,1,2])/2
    k[...,1] = (K[...,0,2] - K[...,2,0])/2
    k[...,2] = (K[...,1,0] - K[...,0,1])/2
    return k

@export
class SO3(LieGroup):
    lie_dim = 3
    rep_dim = 3
    q_dim = 1
    def __init__(self,alpha=.2):
        super().__init__()
        self.alpha = alpha
    
    def exp(self,w):
        """ Rodriguez's formula, assuming shape (*,3)
            where components 1,2,3 are the generators for xrot,yrot,zrot"""
        theta = norm(w,dim=-1)[...,None,None]
        K = cross_matrix(w)
        I = torch.eye(3,device=K.device,dtype=K.dtype)
        Rs = I + K*sinc(theta) + (K@K)*cosc(theta)
        return Rs
    
    def log(self,R):
        """ Computes components in terms of generators rx,ry,rz. Shape (*,3,3)"""
        trR = R[...,0,0]+R[...,1,1]+R[...,2,2]
        costheta = ((trR-1)/2).clamp(max=1,min=-1).unsqueeze(-1)
        theta = torch.acos(costheta)
        logR = uncross_matrix(R)*sinc_inv(theta)
        return logR
    
    def components2matrix(self,a): # a: (*,3)
        return cross_matrix(a)
    
    def matrix2components(self,A): # A: (*,rep_dim,rep_dim)
        return uncross_matrix(A)
    
    def sample(self,*shape,device=torch.device('cuda'),dtype=torch.float32):
        q = torch.randn(*shape,4,device=device,dtype=dtype)
        q /= norm(q,dim=-1).unsqueeze(-1)
        theta_2 = torch.atan2(norm(q[...,1:],dim=-1),q[...,0]).unsqueeze(-1)
        so3_elem = 2*sinc_inv(theta_2)*q[...,1:] # # (sin(x/2)u -> xu) for x angle and u direction
        R = self.exp(so3_elem)
        return R
    
    def lifted_elems(self,pt,nsamples,**kwargs):
        """ Lifting from R^3 -> SO(3) , R^3/SO(3). pt shape (*,3)
            First get a random rotation Rz about [1,0,0] by the appropriate angle
            and then rotate from [1,0,0] to p/\|p\| with Rp  to get RpRz and then
            convert to logarithmic coordinates log(RpRz), \|p\|"""
        d=self.rep_dim
        device,dtype = pt.device,pt.dtype
        # Sample stabilizer of the origin
        q = torch.randn(*pt.shape[:-1],nsamples,4,device=device,dtype=dtype)
        q /= norm(q,dim=-1).unsqueeze(-1)
        theta = 2*torch.atan2(norm(q[...,1:],dim=-1),q[...,0]).unsqueeze(-1)
        zhat = torch.zeros(*pt.shape[:-1],nsamples,3,device=device,dtype=dtype) # (*,3)
        zhat[...,0] = 1#theta
        Rz = self.exp(zhat*theta)

        # Compute the rotation between zhat and p
        r = norm(pt,dim=-1).unsqueeze(-1) # (*,1)
        assert not torch.any(torch.isinf(pt)|torch.isnan(pt))
        p_on_sphere = pt/r.clamp(min=1e-5)
        w = torch.cross(zhat,p_on_sphere[...,None,:].expand(*zhat.shape))
        sin = norm(w,dim=-1)
        cos = p_on_sphere[...,None,0]
        
        angle = torch.atan2(sin,cos).unsqueeze(-1) #cos angle
        Rp = self.exp(w*sinc_inv(angle))
        
        # Combine the rotations into one
        A = self.log(Rp@Rz)  # Convert to lie algebra element
        assert not torch.any(torch.isnan(A)|torch.isinf(A))
        q = r[...,None,:].expand(*r.shape[:-1],nsamples,1) # The orbit identifier is \|x\|
        flat_q = q.reshape(*r.shape[:-2],r.shape[-2]*nsamples,1)
        flat_a = A.reshape(*pt.shape[:-2],pt.shape[-2]*nsamples,d)
        return flat_a, flat_q

@export
class SE3(SO3):
    lie_dim = 6
    rep_dim = 4
    q_dim = 0
    def __init__(self,alpha=.2,per_point=True):
        super().__init__()
        self.alpha = alpha
        self.per_point = per_point

    def exp(self,w):
        theta = norm(w[...,:3],dim=-1)[...,None,None]
        K = cross_matrix(w[...,:3])
        R = super().exp(w[...,:3])
        I = torch.eye(3,device=w.device,dtype=w.dtype)
        V = I + cosc(theta)*K + sincc(theta)*(K@K)
        U = torch.zeros(*w.shape[:-1],4,4,device=w.device,dtype=w.dtype)
        U[...,:3,:3] = R
        U[...,:3,3] = (V@w[...,3:].unsqueeze(-1)).squeeze(-1)
        U[...,3,3] = 1
        return U
    
    def log(self,U):
        w = super().log(U[...,:3,:3])
        I = torch.eye(3,device=w.device,dtype=w.dtype)
        K = cross_matrix(w[...,:3])
        theta = norm(w,dim=-1)[...,None,None]#%(2*np.pi)
        #theta[theta>np.pi] -= 2*np.pi
        cosccc = coscc(theta)
        Vinv = I - K/2 + cosccc*(K@K)
        u = (Vinv@U[...,:3,3].unsqueeze(-1)).squeeze(-1)
        #assert not torch.any(torch.isnan(u)), f"nans in u log {torch.isnan(u).sum()}, {torch.where(torch.isnan(u))}"
        return torch.cat([w,u],dim=-1)

    
    def components2matrix(self,a): # a: (*,3)
        A = torch.zeros(*a.shape[:-1],4,4,device=a.device,dtype=a.dtype)
        A[...,:3,:3] = cross_matrix(a[...,:3])
        A[...,:3,3] = a[...,3:]
        return A
    
    def matrix2components(self,A): # A: (*,4,4)
        return torch.cat([uncross_matrix(A[...,:3,:3]),A[...,:3,3]],dim=-1)

    def lifted_elems(self,pt,nsamples):
        """ pt (bs,n,D) mask (bs,n), per_point specifies whether to
            use a different group element per atom in the molecule"""
        #return farthest_lift(self,pt,mask,nsamples,alpha)
        # same lifts for each point right now
        bs,n = pt.shape[:2]
        if self.per_point:
            q = torch.randn(bs,n,nsamples,4,device=pt.device,dtype=pt.dtype)
        else:
            q = torch.randn(bs,1,nsamples,4,device=pt.device,dtype=pt.dtype)
        q /= norm(q,dim=-1).unsqueeze(-1)
        theta_2 = torch.atan2(norm(q[...,1:],dim=-1),q[...,0]).unsqueeze(-1)
        so3_elem = 2*sinc_inv(theta_2)*q[...,1:] # (sin(x/2)u -> xu) for x angle and u direction
        se3_elem = torch.cat([so3_elem,torch.zeros_like(so3_elem)],dim=-1)
        R = self.exp(se3_elem)
        T = torch.zeros(bs,n,nsamples,4,4,device=pt.device,dtype=pt.dtype) # (bs,n,nsamples,4,4)
        T[...,:,:] = torch.eye(4,device=pt.device,dtype=pt.dtype)
        T[...,:3,3] = pt[:,:,None,:] # (bs,n,1,3)
        a = self.log(T@R)#@R) # bs, n, nsamples, 6
        return a.reshape(bs,n*nsamples,6), None
    def distance(self,abq_pairs):
        dist_rot = norm(abq_pairs[...,:3],dim=-1)
        dist_trans = norm(abq_pairs[...,3:],dim=-1)
        return dist_rot*self.alpha + (1-self.alpha)*dist_trans

@export
class Trivial(LieGroup):
    lie_dim=0
    def __init__(self,dim=2):
        super().__init__()
        self.q_dim = dim
        self.rep_dim = dim

    def lift(self,x,nsamples,**kwargs):
        assert nsamples==1, "Abelian group, no need for nsamples"
        p,v,m = x
        bs,n,d = p.shape
        qa = p[...,:,None,:].expand(bs,n,n,d)
        qb = p[...,None,:,:].expand(bs,n,n,d)
        q = torch.cat([qa,qb],dim=-1)
        return q,v,m
    # def distance(self,abq_pairs):
    #     qa = abq_pairs[...,:self.q_dim]
    #     qb = abq_pairs[...,self.q_dim:]
    #     return norm(qa-qb,dim=-1)

@export
class FakeSchGroup(object,metaclass=Named):
    lie_dim=0
    rep_dim=3
    q_dim=1
    
    def lift(self,x,nsamples,**kwargs):
        """assumes p has shape (*,n,2), vals has shape (*,n,c), mask has shape (*,n)
            returns (a,v) with shapes [(*,n*nsamples,lie_dim),(*,n*nsamples,c)"""
        p,v,m = x
        q = (p[...,:,None,:] - p[...,None,:,:]).norm(dim=-1).unsqueeze(-1)
        return (q,v,m)
    def distance(self,abq_pairs):
        return abq_pairs


